"""复合 Stage-1 流程编排入口。

本文件负责把三个阶段串起来：先让 I3DGS 对无序输入图像进行位姿估计和
Bundle Adjustment（BA），再由 I3DGS 已有的 ``--cglf_scene_path`` 适配器把
注册相机、图像和 BA landmarks 导出为 COLMAP/CGLF 场景，最后调用官方
Scaffold-GS ``train.py`` 进行建模。输入是原始场景目录，输出包括 I3DGS
中间结果、导出的 COLMAP 场景和 Scaffold-GS checkpoint。

这里不实现位姿估计、BA、Gaussian Splatting 或 Scaffold-GS 的训练算法，
只负责路径准备、命令编排、阶段间验证、日志保存和 manifest 记录；旧的
CGLF Stage-1 训练入口也不在本文件的调用链中。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any, Iterable, Sequence


REQUIRED_PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "red",
    "green",
    "blue",
)

# points3D.ply 是 exported_scene 交给 Scaffold-GS 的稀疏几何初始化文件。
# 这些字段既包含坐标和法线，也包含点的颜色；校验时会按名称查找，而不是
# 假设属性顺序，因此可以解释常见的 COLMAP/PLY 写出差异。
# PLY scalar type names, including the aliases used by the common PLY writers.
_PLY_SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def _absolute_path(value: os.PathLike[str] | str) -> Path:
    """把输入路径解析为绝对路径，但不创建任何目录。

    参数 ``value`` 可以是字符串或 ``PathLike``。绝对路径让 I3DGS 和
    Scaffold-GS 即使使用不同的 ``cwd``，也不会对同一个相对路径产生不同
    解释。返回值用于命令参数和 manifest；该函数只读取路径，不检查其存在性。
    """

    return Path(value).expanduser().resolve(strict=False)


def _path_key(path: Path) -> str:
    """生成用于路径关系比较的规范字符串。

    Windows 文件系统通常不区分大小写，因此这里结合 ``resolve`` 和
    ``os.path.normcase``，使大小写差异、已存在的符号链接目标都参与同一套
    父子目录判断。该函数不会创建或修改路径。
    """

    return os.path.normcase(str(path.resolve(strict=False)))


def _same_or_within(candidate: Path, directory: Path) -> bool:
    """判断 ``candidate`` 是否等于 ``directory`` 或位于其内部。

    参数是两个待比较的路径；返回布尔值。不同 Windows 盘符之间没有共同
    父目录时返回 ``False``。该判断用于保护输入数据和输出目录，不产生文件
    系统副作用。
    """

    candidate_key = _path_key(candidate)
    directory_key = _path_key(directory)
    try:
        return os.path.commonpath((candidate_key, directory_key)) == directory_key
    except ValueError:
        # Different Windows drives cannot overlap.
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    """判断两个路径是否相等或任意一方是另一方的父目录。

    这不是文件内容检查，而是对三个输出目录做拓扑关系检查，防止一个阶段
    的输出落入另一个阶段的目录。返回布尔值，不修改文件系统。
    """

    return _same_or_within(first, second) or _same_or_within(second, first)


def _directory_has_entries(path: Path) -> bool:
    """判断已有目录是否包含至少一个目录项。

    目录不存在时返回 ``False``；目录存在时只遍历其直接子项。该结果用于
    决定训练程序是否可能覆盖已有输出，不会创建、删除或写入目录。
    """

    try:
        return any(path.iterdir())
    except FileNotFoundError:
        return False


def _validate_input_and_outputs(
    input_scene: Path,
    i3dgs_output: Path,
    exported_scene: Path,
    scaffold_output: Path,
    *,
    skip_i3dgs: bool,
) -> None:
    """在启动任何训练前执行输入、输出和模式门禁。

    参数包含原始输入场景、I3DGS 输出、适配器导出场景、Scaffold-GS 输出
    和 ``skip_i3dgs`` 开关。函数没有返回值；路径不满足约束时抛出
    ``FileNotFoundError``、``FileExistsError`` 或 ``ValueError``。它只读文件
    系统状态，不创建目录，也不启动子进程，对应 Stage-1 的准备步骤。
    """

    if not input_scene.is_dir():
        raise FileNotFoundError(f"input_scene is not an existing directory: {input_scene}")

    outputs = (i3dgs_output, exported_scene, scaffold_output)
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :]:
            if _paths_overlap(first, second):
                raise ValueError(
                    "Output paths must not be equal or nested: "
                    f"{first} <-> {second}"
                )
        if _same_or_within(first, input_scene):
            raise ValueError(
                f"Output path must not be inside input_scene: {first} "
                f"(input_scene={input_scene})"
            )

    if skip_i3dgs:
        if not exported_scene.is_dir():
            raise FileNotFoundError(
                "--skip_i3dgs requires an existing exported_scene directory: "
                f"{exported_scene}"
            )
    elif os.path.lexists(exported_scene):
        # 适配器本身也拒绝覆盖已有导出目录；这里提前检查，避免 I3DGS 已经
        # 花费较长时间后才因目标目录存在而失败。
        raise FileExistsError(
            f"Normal mode requires exported_scene not to exist: {exported_scene}"
        )

    output_dirs = [("scaffold_output", scaffold_output)]
    if not skip_i3dgs:
        output_dirs.insert(0, ("i3dgs_output", i3dgs_output))
    for label, path in output_dirs:
        if not os.path.lexists(path):
            continue
        if os.path.islink(path) or not path.is_dir():
            raise FileExistsError(
                f"Refusing to use existing non-directory output {label}: {path}"
            )
        if _directory_has_entries(path):
            raise FileExistsError(
                f"Refusing to overwrite non-empty {label}: {path}"
            )


def _parse_ply_header(handle: Any) -> tuple[str, list[dict[str, Any]], int]:
    """读取 PLY header，并返回格式、element 定义和数据区偏移。

    参数 ``handle`` 必须是以二进制模式打开的 PLY 文件句柄。返回值包含
    ``ascii`` 或 ``binary_little_endian`` 格式名、每个 element 的数量及属性
    描述、以及 header 结束后的当前位置。遇到缺失 ``end_header``、未知类型
    或不支持的格式时抛出 ``ValueError``；只读文件，不写文件。
    """

    header_lines: list[str] = []
    while True:
        raw_line = handle.readline()
        if not raw_line:
            raise ValueError("PLY header is missing end_header")
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header is not ASCII") from exc
        header_lines.append(line)
        if line == "end_header":
            break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("PLY must start with 'ply'")

    ply_format: str | None = None
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    # PLY header 用 element 描述记录数量，用 property 描述每条记录的字段。
    # 对 points3D.ply 而言，vertex 是点，x/y/z 是位置，nx/ny/nz 是法线，
    # red/green/blue 是颜色；这些定义决定后续如何解释二进制字节。
    for line in header_lines[1:]:
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue
        parts = line.split()
        if parts[0] == "format" and len(parts) >= 2:
            ply_format = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise ValueError(f"Invalid PLY element count: {line}") from exc
            if count < 0:
                raise ValueError(f"Negative PLY element count: {line}")
            current = {"name": parts[1], "count": count, "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if len(parts) == 3:
                property_type, name = parts[1], parts[2]
                if property_type not in _PLY_SCALAR_TYPES:
                    raise ValueError(f"Unsupported PLY scalar type: {property_type}")
                current["properties"].append(
                    {"kind": "scalar", "type": property_type, "name": name}
                )
            elif len(parts) == 5 and parts[1] == "list":
                count_type, item_type, name = parts[2], parts[3], parts[4]
                if count_type not in _PLY_SCALAR_TYPES or item_type not in _PLY_SCALAR_TYPES:
                    raise ValueError(f"Unsupported PLY list type: {line}")
                current["properties"].append(
                    {
                        "kind": "list",
                        "count_type": count_type,
                        "item_type": item_type,
                        "name": name,
                    }
                )
            else:
                raise ValueError(f"Invalid PLY property declaration: {line}")

    if ply_format is None:
        raise ValueError("PLY format declaration is missing")
    if ply_format not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format: {ply_format}")
    return ply_format, elements, handle.tell()


def _scalar_from_bytes(data: bytes, offset: int, type_name: str) -> tuple[Any, int]:
    """从 PLY 二进制 body 的指定偏移读取一个小端标量。

    参数 ``data`` 是文件 body，``offset`` 是当前游标，``type_name`` 是 PLY
    类型名称。返回值是解码后的标量和下一游标；数据不足时抛出
    ``ValueError``。函数只处理内存数据，不产生文件或进程副作用。
    """

    fmt, size = _PLY_SCALAR_TYPES[type_name]
    end = offset + size
    if end > len(data):
        raise ValueError("PLY binary body is shorter than its header declares")
    return struct.unpack_from("<" + fmt, data, offset)[0], end


def _read_binary_record(
    data: bytes, offset: int, properties: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    """按属性定义读取一条 binary little-endian PLY 记录。

    参数 ``properties`` 来自 PLY header；函数逐个读取 scalar 或 list 属性，
    返回属性名到数值的字典及下一条记录的偏移。记录不完整或 list 数量非法
    时抛出 ``ValueError``。只读内存，不写文件。
    """

    values: dict[str, Any] = {}
    for prop in properties:
        if prop["kind"] == "scalar":
            value, offset = _scalar_from_bytes(data, offset, prop["type"])
        else:
            count, offset = _scalar_from_bytes(data, offset, prop["count_type"])
            if not isinstance(count, int) or count < 0:
                raise ValueError("PLY list property has an invalid count")
            value = []
            for _ in range(count):
                item, offset = _scalar_from_bytes(data, offset, prop["item_type"])
                value.append(item)
        values[prop["name"]] = value
    return values, offset


def _read_ascii_record(tokens: Sequence[str], offset: int, properties: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """按属性定义读取一条 ASCII PLY 记录。

    参数 ``tokens`` 是 PLY body 按空白拆分后的文本，``offset`` 是当前 token
    位置，``properties`` 是 header 中的属性定义。返回属性字典和下一位置；
    文本不足、数值或 list 数量无法解析时抛出 ``ValueError``。函数只解析
    内存中的文本，不创建文件。
    """

    values: dict[str, Any] = {}
    for prop in properties:
        if offset >= len(tokens):
            raise ValueError("PLY ASCII body is shorter than its header declares")
        if prop["kind"] == "scalar":
            token = tokens[offset]
            offset += 1
            try:
                value: Any = float(token)
                if prop["type"] not in {"float", "float32", "double", "float64"}:
                    value = int(float(token))
            except ValueError as exc:
                raise ValueError(f"Invalid PLY ASCII scalar: {token}") from exc
        else:
            try:
                count = int(tokens[offset])
            except ValueError as exc:
                raise ValueError(f"Invalid PLY ASCII list count: {tokens[offset]}") from exc
            offset += 1
            if count < 0 or offset + count > len(tokens):
                raise ValueError("PLY ASCII list property has an invalid count")
            value = tokens[offset : offset + count]
            offset += count
        values[prop["name"]] = value
    return values, offset


def _validate_points3d_ply(path: Path) -> dict[str, Any]:
    """验证 points3D.ply 的 vertex 字段、数量和有限 XYZ。

    参数 ``path`` 是待验证的 PLY 路径。返回顶点数量和属性名列表；缺少
    ``vertex``、顶点为空、缺少必需字段、数据区不完整或 XYZ 含非有限值时
    抛出 ``ValueError``。函数支持 ASCII 和 binary little-endian 两种读取路径，
    只读 PLY，不写文件；它对应 exported_scene 验证步骤。
    """

    with path.open("rb") as handle:
        ply_format, elements, body_offset = _parse_ply_header(handle)
        body = handle.read()

    vertex_index = next(
        (index for index, element in enumerate(elements) if element["name"] == "vertex"),
        None,
    )
    if vertex_index is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = elements[vertex_index]
    vertex_count = int(vertex["count"])
    if vertex_count <= 0:
        raise ValueError(f"PLY must contain at least one vertex: {path}")
    properties = vertex["properties"]
    property_names = [prop["name"] for prop in properties]
    # 坐标决定 landmarks 的空间位置，法线和颜色是 Scaffold-GS 读取的
    # 点属性；这里按字段名确认它们都存在，而不是仅凭文件扩展名判断。
    missing = [name for name in REQUIRED_PLY_PROPERTIES if name not in property_names]
    if missing:
        raise ValueError(f"PLY is missing required properties {missing}: {path}")
    if len(set(property_names)) != len(property_names):
        raise ValueError(f"PLY vertex properties contain duplicate names: {path}")

    xyz_values: list[tuple[float, float, float]] = []
    if ply_format == "binary_little_endian":
        # 适配器写出的 vertex 通常是第一个 element，且字段都是 scalar。此时
        # 可以直接按结构化 dtype 批量读取，避免对大型点云逐字段调用 struct。
        # 如果实际 header 不满足这个布局，下面的通用逐记录路径仍会读取它。
        if vertex_index == 0 and all(prop["kind"] == "scalar" for prop in properties):
            try:
                import numpy as np  # type: ignore
            except ImportError:
                np = None
            if np is not None:
                dtype = np.dtype(
                    [
                        (prop["name"], "<" + _PLY_SCALAR_TYPES[prop["type"]][0])
                        for prop in properties
                    ]
                )
                expected_size = dtype.itemsize * vertex_count
                if len(body) < expected_size:
                    raise ValueError("PLY binary body is shorter than its header declares")
                records = np.frombuffer(body, dtype=dtype, count=vertex_count)
                xyz = np.column_stack([records[name] for name in ("x", "y", "z")])
                # NaN/Inf 不能代表有限的三维位置；将其传给下游会使几何
                # 初始化失去明确含义，因此需要对所有 XYZ 一次性检查。
                if not np.isfinite(xyz).all():
                    raise ValueError(f"PLY XYZ contains non-finite values: {path}")
                return {"vertex_count": vertex_count, "properties": property_names}

        offset = 0
        for element in elements[:vertex_index]:
            for _ in range(int(element["count"])):
                _, offset = _read_binary_record(body, offset, element["properties"])
        for _ in range(vertex_count):
            values, offset = _read_binary_record(body, offset, properties)
            xyz_values.append(
                (float(values["x"]), float(values["y"]), float(values["z"]))
            )
    else:
        try:
            tokens = body.decode("ascii").split()
        except UnicodeDecodeError as exc:
            raise ValueError("PLY ASCII body is not ASCII") from exc
        offset = 0
        for element in elements:
            for _ in range(int(element["count"])):
                values, offset = _read_ascii_record(tokens, offset, element["properties"])
                if element is vertex:
                    xyz_values.append(
                        (float(values["x"]), float(values["y"]), float(values["z"]))
                    )

    # 通用解析路径也必须检查每一个顶点的 XYZ，不能只检查第一个点。
    if not xyz_values or not all(math.isfinite(value) for row in xyz_values for value in row):
        raise ValueError(f"PLY XYZ contains non-finite values: {path}")
    return {"vertex_count": vertex_count, "properties": property_names}


def validate_exported_scene(scene_path: os.PathLike[str] | str) -> dict[str, Any]:
    """验证 I3DGS 适配器生成的 COLMAP/CGLF 场景目录。

    参数 ``scene_path`` 是 exported_scene 根目录。函数检查非空
    ``images/``、``sparse/0/cameras.bin``、``images.bin`` 和 ``points3D.ply``，
    并委托 ``_validate_points3d_ply`` 检查 PLY。返回图像数、顶点数和字段名；
    缺失或为空时抛出 ``FileNotFoundError``/``ValueError``。只读文件系统，
    对应 Stage-1 中 I3DGS 导出之后、Scaffold-GS 启动之前的验证步骤。
    """

    scene = _absolute_path(scene_path)
    # exported_scene 是 I3DGS 到 Scaffold-GS 之间的接口：images/ 提供实际
    # 训练图像，cameras.bin 保存相机内参，images.bin 保存注册图像的外参和
    # 观测关系，points3D.ply 则提供稀疏 3D landmarks 作为建模初始化。
    images_dir = scene / "images"
    sparse_dir = scene / "sparse" / "0"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Exported scene images/ is missing: {images_dir}")
    image_files = [
        entry
        for entry in images_dir.iterdir()
        if entry.is_file() and entry.stat().st_size > 0
    ]
    if not image_files:
        raise ValueError(f"Exported scene images/ is empty: {images_dir}")

    required_files = {
        "cameras_bin": sparse_dir / "cameras.bin",
        "images_bin": sparse_dir / "images.bin",
        "points3D_ply": sparse_dir / "points3D.ply",
    }
    for label, path in required_files.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Exported scene {label} is missing or empty: {path}")
    # 仅检查三个 COLMAP 文件存在还不足以说明点云可被读取，因此把 PLY
    # header、vertex 字段、顶点数量和有限 XYZ 的检查集中交给专门函数。
    ply_info = _validate_points3d_ply(required_files["points3D_ply"])
    return {
        "image_count": len(image_files),
        **ply_info,
    }


def validate_scaffold_checkpoint(
    output_path: os.PathLike[str] | str,
    iterations: int,
    appearance_dim: int,
) -> dict[str, Any]:
    """验证官方 Scaffold-GS 在指定迭代数写出的 checkpoint 文件。

    参数 ``output_path`` 是 Scaffold-GS 输出根目录，``iterations`` 用来定位
    ``point_cloud/iteration_<iterations>/``，``appearance_dim`` 决定是否额外
    要求 ``embedding_appearance.pt``。返回迭代数和检查过的文件列表；缺失或
    空文件时抛出 ``FileNotFoundError``。函数只读目录，对应 Stage-1 的最终
    输出验证步骤。
    """

    output = _absolute_path(output_path)
    iteration_dir = output / "point_cloud" / f"iteration_{iterations}"
    # cfg_args 保存训练配置；point_cloud.ply 保存 Scaffold-GS 的 anchor/点云，
    # 三个 *_mlp.pt 保存 opacity、covariance 和 color 的 MLP 参数。appearance
    # embedding 只有在启用外观维度时才属于预期输出。
    required = [
        output / "cfg_args",
        iteration_dir / "point_cloud.ply",
        iteration_dir / "opacity_mlp.pt",
        iteration_dir / "cov_mlp.pt",
        iteration_dir / "color_mlp.pt",
    ]
    if appearance_dim > 0:
        required.append(iteration_dir / "embedding_appearance.pt")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(
            "Scaffold-GS checkpoint is incomplete; missing or empty files: "
            + ", ".join(missing)
        )
    return {"iteration": iterations, "files": [str(path) for path in required]}


def _resolve_executable(value: str) -> str:
    """把 Python 解释器名称解析成命令数组可使用的绝对路径。

    参数 ``value`` 可以是解释器绝对路径或 PATH 中的名称。返回解析后的
    字符串；不会创建文件，也不会启动解释器。它在构造两个训练命令前执行，
    使两个阶段可以分别使用各自的 Python 环境。
    """

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve(strict=False))
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve(strict=False))
    return str(candidate.resolve(strict=False))


def _build_commands(
    *,
    repo_root: Path,
    input_scene: Path,
    i3dgs_output: Path,
    exported_scene: Path,
    scaffold_output: Path,
    i3dgs_python: str,
    scaffold_python: str,
    iterations: int,
    voxel_size: float,
    appearance_dim: int,
    eval_mode: bool,
    i3dgs_num_iterations: int | None,
) -> tuple[list[str], list[str]]:
    """构造 I3DGS 和 Scaffold-GS 的参数列表。

    参数包含四个场景/输出路径、两个 Python 解释器和训练超参数。返回值是
    ``(i3dgs_command, scaffold_command)`` 两个列表；函数只构造数据，不启动
    子进程，也不创建输出。I3DGS 列表把 ``source_path``、``model_path`` 和
    ``cglf_scene_path`` 传给现有入口，Scaffold-GS 列表把导出的场景和自己的
    迭代、voxel、appearance 参数传给官方入口。
    """

    # 使用参数列表而不是拼接命令字符串，避免路径中的空格被错误拆分，
    # 同时不需要启用 shell=True。两个脚本使用不同的绝对解释器路径，
    # 以便 I3DGS 与 Scaffold-GS 可以分别绑定各自的 Conda 环境。
    i3dgs_command = [
        i3dgs_python,  # I3DGS 环境的 Python 解释器。
        str(repo_root / "i3dgs" / "train.py"),  # I3DGS 位姿/BA 入口。
        "--source_path",  # I3DGS 读取原始无序输入图像的目录。
        str(input_scene),
        "--model_path",  # I3DGS 保存中间重建结果的目录。
        str(i3dgs_output),
        "--cglf_scene_path",  # 让 I3DGS 最终保存后调用已有 CGLF 导出适配器。
        str(exported_scene),
    ]
    if i3dgs_num_iterations is not None:
        # 该可选参数只控制 I3DGS 的 --num_iterations；顶层 --iterations
        # 则专门传给 Scaffold-GS，避免两个项目的同名概念混用。
        i3dgs_command.extend(["--num_iterations", str(i3dgs_num_iterations)])

    scaffold_command = [
        scaffold_python,  # Scaffold-GS 环境的 Python 解释器。
        str(repo_root / "Scaffold-GS" / "train.py"),  # 官方建模入口。
        "--source_path",  # Scaffold-GS 读取 I3DGS 导出的 COLMAP 场景。
        str(exported_scene),
        "--model_path",  # Scaffold-GS 写入 point_cloud 和 MLP checkpoint。
        str(scaffold_output),
        "--iterations",  # Scaffold-GS 的训练总迭代数。
        str(iterations),
        "--save_iterations",  # 请求在目标迭代保存点云。
        str(iterations),
        "--test_iterations",  # 请求在目标迭代执行测试/评估。
        str(iterations),
        "--voxel_size",  # Scaffold-GS 的体素初始化尺度。
        str(voxel_size),
        "--appearance_dim",  # 外观 embedding 维度，并影响 checkpoint 文件集合。
        str(appearance_dim),
    ]
    # eval 只属于 Scaffold-GS 的数据划分和评估流程，不传给负责位姿估计
    # 与 BA 的 I3DGS。关闭时不追加参数，保持官方 Scaffold-GS 的默认行为。
    if eval_mode:
        scaffold_command.append("--eval")
    return i3dgs_command, scaffold_command


def _format_command(command: Sequence[str]) -> str:
    """将参数列表格式化为适合终端和日志阅读的一行命令。

    参数 ``command`` 是待执行的字符串序列；返回值仅用于显示，不会改变
    原始列表，也不会启动子进程。Windows 和 POSIX 使用各自的转义规则。
    """

    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    import shlex

    return shlex.join(list(command))


def _run_logged_command(command: Sequence[str], cwd: Path, log_path: Path, label: str) -> None:
    """在指定项目工作目录运行一个命令并保存合并日志。

    参数 ``command`` 是 subprocess 参数列表，``cwd`` 是对应项目目录，
    ``log_path`` 是该阶段独立日志，``label`` 用于错误信息。函数不返回值；
    子进程非零退出码时抛出 ``RuntimeError``。它会创建日志父目录、创建或
    覆盖日志文件并启动一个子进程；stdout 和 stderr 合并写入同一日志。
    这是 Stage-1 实际调用 I3DGS/Scaffold-GS 的执行步骤。
    """

    # 先写入 cwd 和完整参数列表，后续可以把日志与 manifest 中的命令对应。
    # stderr 使用 STDOUT 合并到同一个句柄；shell=False 保持每个参数边界。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"cwd: {cwd}\n")
        log_handle.write(f"command: {_format_command(command)}\n\n")
        log_handle.flush()
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}; see log {log_path}"
        )


def _git_sha(repository: Path) -> str:
    """读取指定仓库当前检出的 Git SHA。

    参数 ``repository`` 是 I3DGS 或 Scaffold-GS 仓库目录。成功时返回
    ``git rev-parse HEAD`` 的文本；Git 不可用或查询失败时返回 ``unknown``。
    函数启动一个只读 Git 子进程，不修改仓库，对应 manifest 版本追踪步骤。
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"


def _write_manifest(
    scaffold_output: Path,
    *,
    input_scene: Path,
    i3dgs_output: Path,
    exported_scene: Path,
    i3dgs_command: Sequence[str],
    scaffold_command: Sequence[str],
    skip_i3dgs: bool,
    iterations: int,
    voxel_size: float,
    appearance_dim: int,
    eval_mode: bool,
    i3dgs_sha: str,
    scaffold_sha: str,
    i3dgs_log: Path,
    scaffold_log: Path,
) -> Path:
    """将一次 Stage-1 运行的路径、命令、参数和版本写入 manifest。

    ``scaffold_output`` 是 manifest 所在的输出根目录；其余参数记录输入、
    两个输出、两个命令数组、skip 状态、训练参数、Git SHA 和日志路径。返回
    ``stage1_manifest.json`` 路径。函数会创建或覆盖该 JSON 文件，不启动子
    进程；调用顺序位于 checkpoint 验证之后。
    """

    manifest_path = scaffold_output / "stage1_manifest.json"
    manifest = {
        "schema_version": 1,
        "input_scene": str(input_scene),
        "i3dgs_output": str(i3dgs_output),
        "exported_scene": str(exported_scene),
        "scaffold_output": str(scaffold_output),
        "iterations": iterations,
        "voxel_size": voxel_size,
        "appearance_dim": appearance_dim,
        "eval": eval_mode,
        "commands": {
            "i3dgs": {
                "status": "skipped" if skip_i3dgs else "executed",
                "argv": [] if skip_i3dgs else list(i3dgs_command),
            },
            "scaffold_gs": {"status": "executed", "argv": list(scaffold_command)},
        },
        # 下面的别名保留两组直接可读取的命令数组；structured entries 另外
        # 携带执行状态，因此 skip 模式可以同时表达“没有执行 I3DGS”。
        "i3dgs_command": [] if skip_i3dgs else list(i3dgs_command),
        "scaffold_command": list(scaffold_command),
        "i3dgs_git_sha": i3dgs_sha,
        "scaffold_gs_git_sha": scaffold_sha,
        "git": {"i3dgs": i3dgs_sha, "scaffold_gs": scaffold_sha},
        "logs": {"i3dgs": str(i3dgs_log), "scaffold_gs": str(scaffold_log)},
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    """创建复合 Stage-1 的命令行参数解析器。

    返回 ``ArgumentParser``，供 ``main`` 解析输入场景、三个输出路径、两个
    Python 环境、训练参数以及 skip/dry-run 开关。该函数只定义参数，不创建
    文件、目录或子进程，对应 Stage-1 的入口配置步骤。
    """

    parser = argparse.ArgumentParser(
        description="I3DGS pose/BA + CGLF export + official Scaffold-GS Stage-1"
    )
    parser.add_argument("--input_scene", required=True, help="原始输入场景目录")
    parser.add_argument("--i3dgs_output", required=True, help="I3DGS 输出目录")
    parser.add_argument("--exported_scene", required=True, help="适配器导出的 COLMAP 场景")
    parser.add_argument("--scaffold_output", required=True, help="Scaffold-GS 输出目录")
    parser.add_argument("--i3dgs_python", default=sys.executable, help="I3DGS 使用的 Python")
    parser.add_argument("--scaffold_python", default=sys.executable, help="Scaffold-GS 使用的 Python")
    parser.add_argument("--iterations", type=int, default=30_000, help="仅传给 Scaffold-GS 的迭代数")
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--appearance_dim", type=int, default=32)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="仅传给 Scaffold-GS：启用训练/测试视角划分和测试迭代评估",
    )
    parser.add_argument(
        "--i3dgs_num_iterations",
        type=int,
        default=None,
        help="可选：传给 I3DGS 的 --num_iterations；不设置则使用 I3DGS 默认值",
    )
    parser.add_argument("--skip_i3dgs", action="store_true", help="复用已验证的 exported_scene")
    parser.add_argument("--dry_run", action="store_true", help="仅打印命令，不训练、不创建输出")
    return parser


def run_pipeline(namespace: argparse.Namespace) -> dict[str, Any] | None:
    """按顺序执行 I3DGS、场景导出验证和 Scaffold-GS Stage-1。

    参数 ``namespace`` 是 ``build_parser`` 产生的参数对象。正常模式返回
    manifest 与两个日志路径；dry-run 返回 ``None``。函数可能抛出路径、参数、
    子进程和输出验证相关异常。正常模式会创建日志并启动最多两个训练子进程，
    skip 模式只启动 Scaffold-GS，dry-run 不创建真实输出。它是本文件的主
    Stage-1 控制流。
    """

    repo_root = Path(__file__).resolve().parents[1]
    # 三个场景路径先统一为绝对路径：I3DGS 和 Scaffold-GS 会在不同 cwd 下
    # 运行，绝对路径让命令参数和最后写入的 manifest 指向同一份数据。
    input_scene = _absolute_path(namespace.input_scene)
    i3dgs_output = _absolute_path(namespace.i3dgs_output)
    exported_scene = _absolute_path(namespace.exported_scene)
    scaffold_output = _absolute_path(namespace.scaffold_output)
    # 在启动外部程序前检查数值参数，避免把无法解释的训练配置传入子进程。
    if namespace.iterations <= 0:
        raise ValueError("iterations must be positive")
    if namespace.appearance_dim < 0:
        raise ValueError("appearance_dim must be non-negative")
    if not math.isfinite(namespace.voxel_size):
        raise ValueError("voxel_size must be finite")
    if namespace.i3dgs_num_iterations is not None and namespace.i3dgs_num_iterations <= 0:
        raise ValueError("i3dgs_num_iterations must be positive")

    _validate_input_and_outputs(
        input_scene,
        i3dgs_output,
        exported_scene,
        scaffold_output,
        skip_i3dgs=namespace.skip_i3dgs,
    )
    i3dgs_python = _resolve_executable(namespace.i3dgs_python)
    scaffold_python = _resolve_executable(namespace.scaffold_python)
    i3dgs_command, scaffold_command = _build_commands(
        repo_root=repo_root,
        input_scene=input_scene,
        i3dgs_output=i3dgs_output,
        exported_scene=exported_scene,
        scaffold_output=scaffold_output,
        i3dgs_python=i3dgs_python,
        scaffold_python=scaffold_python,
        iterations=namespace.iterations,
        voxel_size=namespace.voxel_size,
        appearance_dim=namespace.appearance_dim,
        eval_mode=namespace.eval,
        i3dgs_num_iterations=namespace.i3dgs_num_iterations,
    )

    if namespace.skip_i3dgs:
        # skip 不是跳过所有检查，而是复用一个已经存在的 exported_scene；
        # 只有目录结构、文件和 PLY 都通过验证后，才允许进入 Scaffold-GS。
        validate_exported_scene(exported_scene)

    if namespace.dry_run:
        # dry-run 的边界在这里返回：前面的路径检查仍然是只读的，但不会
        # 创建日志/输出目录、读取 Git SHA，也不会调用任一训练脚本。
        if not namespace.skip_i3dgs:
            print(f"I3DGS command: {_format_command(i3dgs_command)}")
        print(f"Scaffold-GS command: {_format_command(scaffold_command)}")
        return None

    i3dgs_log = i3dgs_output.parent / f"{i3dgs_output.name}.i3dgs.log"
    scaffold_log = scaffold_output.parent / f"{scaffold_output.name}.scaffold-gs.log"
    if not namespace.skip_i3dgs:
        # I3DGS 的一次运行同时负责位姿、BA 和最终 adapter export；其成功
        # 返回后才检查 exported_scene，确保下游不会读取半成品目录。
        _run_logged_command(i3dgs_command, repo_root / "i3dgs", i3dgs_log, "I3DGS")
        validate_exported_scene(exported_scene)
    # exported_scene 验证通过（或 skip 模式复用已验证场景）后，才把它作为
    # Scaffold-GS 的 source_path。两个脚本使用各自项目目录作为 cwd。
    _run_logged_command(
        scaffold_command,
        repo_root / "Scaffold-GS",
        scaffold_log,
        "Scaffold-GS",
    )
    validate_scaffold_checkpoint(
        scaffold_output, namespace.iterations, namespace.appearance_dim
    )

    # The pass may be skipped, but the manifest still records the checked-out
    # I3DGS revision; the command status separately records that it was skipped.
    i3dgs_sha = _git_sha(repo_root / "i3dgs")
    scaffold_sha = _git_sha(repo_root / "Scaffold-GS")
    manifest_path = _write_manifest(
        scaffold_output,
        input_scene=input_scene,
        i3dgs_output=i3dgs_output,
        exported_scene=exported_scene,
        i3dgs_command=i3dgs_command,
        scaffold_command=scaffold_command,
        skip_i3dgs=namespace.skip_i3dgs,
        iterations=namespace.iterations,
        voxel_size=namespace.voxel_size,
        appearance_dim=namespace.appearance_dim,
        eval_mode=namespace.eval,
        i3dgs_sha=i3dgs_sha,
        scaffold_sha=scaffold_sha,
        i3dgs_log=i3dgs_log,
        scaffold_log=scaffold_log,
    )
    print(f"Stage-1 complete; manifest: {manifest_path}")
    return {
        "manifest": str(manifest_path),
        "i3dgs_log": str(i3dgs_log),
        "scaffold_log": str(scaffold_log),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行并把流程结果转换为进程退出码。

    参数 ``argv`` 可用于测试时传入自定义参数；为 ``None`` 时使用真实命令行。
    成功返回 ``0``，路径、配置、子进程或输出验证异常返回 ``1``。函数本身
    不改变训练逻辑，真正的目录和子进程副作用由 ``run_pipeline`` 的分支决定。
    """

    args = build_parser().parse_args(argv)
    try:
        run_pipeline(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Stage-1 pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
