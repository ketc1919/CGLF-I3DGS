Immediate3DGS License  
===========================  

**Inria** holds all the ownership rights on the *Software* named  
**Immediate3DGS**. The method implemented by the *Software* is patent-pending  
and owned by Inria.  

The *Software* is in the process of being registered with the Agence pour la  
Protection des Programmes (APP).  

The *Software* is still being developed by the *Licensor*.  

*Licensor*'s goal is to allow the research community to use, test and evaluate  
the *Software*.  

In cases where the constraints of the License prevent you from using the  
*Software*, you can contact OnTheFly (https://onthefly3d.com) for commercial  
opportunities.  

## 1.  Definitions  

*Licensee* means any person or entity that uses the *Software* and distributes  
its *Work*.  

*Licensor* means the owners of the *Software*, i.e. Inria.  

*Software* means the original work of authorship made available under this  
License i.e. Immediate3DGS.  

*Work* means the *Software* and any additions to or derivative works of the  
*Software* that are made available under this License.  

## 2.  Purpose  
This license is intended to define the rights granted to the *Licensee* by  
Licensors under the *Software*.  

## 3.  Rights granted  

For the above reasons Licensors have decided to distribute the *Software*.  
Licensors grant non-exclusive rights to use the *Software* for research purposes  
to research users (both academic and industrial), free of charge, without right  
to sublicense. The *Software* may be used "non-commercially", i.e., for research  
and/or evaluation purposes only.  

Subject to the terms and conditions of this License, you are granted a  
non-exclusive, royalty-free, license to reproduce, prepare derivative works of,  
publicly display, publicly perform and distribute its *Work* and any resulting  
derivative works in any form.  

## 4.  Limitations  

**4.1 Redistribution.** You may reproduce or distribute the *Work* only if (a)  
you do so under this License, (b) you include a complete copy of this License  
with your distribution, and (c) you retain without modification any copyright,  
patent, trademark, or attribution notices that are present in the *Work*.  

**4.2 Derivative Works.** You may specify that additional or different terms  
apply to the use, reproduction, and distribution of your derivative works of the  
*Work* ("Your Terms") only if (a) Your Terms provide that the use limitation in  
Section 3 applies to your derivative works, and (b) you identify the specific  
derivative works that are subject to Your Terms. Notwithstanding Your Terms,  
this License (including the redistribution requirements in Section 4.1) will  
continue to apply to the *Work* itself.  

**4.3** Any other use without prior consent of Licensors is prohibited. Research  
users explicitly acknowledge having received from Licensors all information  
allowing to appreciate the adequacy between the *Software* and their needs and  
to undertake all necessary precautions for its execution and use.  

**4.4** The *Software* is provided both as a compiled library file and as source  
code. In case of using the *Software* for a publication or other results  
obtained through the use of the *Software*, users are strongly encouraged to  
cite the corresponding publications as explained in the documentation of the  
*Software*.  

## 5.  Disclaimer  

THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE *SOFTWARE* FOR COMMERCIAL  
PURPOSES. ANY SUCH ACTION WILL CONSTITUTE A FORGERY. YOU MUST CONTACT INRIA FOR  
ANY UNAUTHORIZED USE: stip-sophia.transfert@inria.fr.  THIS *SOFTWARE* IS  
PROVIDED "AS IS" WITHOUT ANY WARRANTIES OF ANY NATURE AND ANY EXPRESS OR IMPLIED  
WARRANTIES, WITH REGARDS TO COMMERCIAL USE, PROFESSIONAL USE, LEGAL OR NOT, OR  
OTHER, OR COMMERCIALISATION OR ADAPTATION. UNLESS EXPLICITLY PROVIDED BY LAW, IN  
NO EVENT, SHALL INRIA OR THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,  
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT  
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA, OR  
PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,  
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR  
OTHERWISE) ARISING FROM, OUT OF OR IN CONNECTION WITH THE *SOFTWARE* OR THE USE  
OR OTHER DEALINGS IN THE *SOFTWARE*.  

## 6. Files subject to permissive licenses  
Some files contain code adapted from third-party projects distributed under more  
permissive licenses, the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)  
(a copy is provided in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)) unless  
stated otherwise below; the adapted portions remain governed by their original  
licenses and are marked as such in the corresponding files:  
- `poses/ransac.cu`: adapted from 
  [OpenCV](https://github.com/opencv/opencv) (usac module).  
- `poses/p3p.cu`: CUDA port of the Lambda Twist P3P solver from  
  [lambdatwist](https://github.com/vlarsson/lambdatwist), under the  
  BSD-3-Clause license (retained in the header of the file).  
- `poses/lighterglue_matcher.py`: adapted from  
  [accelerated_features](https://github.com/verlab/accelerated_features) and  
  [Kornia](https://github.com/kornia/kornia) (LightGlue).  
- `scene/extractor_model.py` and parts of `poses/feature_detector.py`: adapted  
  from [accelerated_features](https://github.com/verlab/accelerated_features)  
  (XFeat).  
- Parts of `poses/depth_anything3.py`: adapted from  
  [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (input  
  processing).  
- `dataloaders/read_write_model.py`: from  
  [COLMAP](https://github.com/colmap/colmap), under the BSD-3-Clause license  
  (retained at the top of the file).  
