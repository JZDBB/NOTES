# 比较经验

## yolo v5 vs Faster RCNN

### **框架选择**

- **使用 YOLO-v5 进行推理**

第一步就是克隆 YOLO-v5 的 repo，并安装所有的依赖要求，按照以下方法下载不同预训练 COCO 模型的所有权重：

```shell
bash weights/download_weights.sh
```

要对视频进行推理，就必须将传递给视频的路径以及要使用的模型的权重。如果没有设置权重参数，那么在默认情况下，代码在 YOLO 小模型上运行。

```shell
python detect.py --source video/MOT20-01-raw-cut1.mp4 --output video_out/ --weights weights/yolov5s.pt --conf-thres 0.4
```

输出视频将保存在输出文件夹中。

- **Faster RCNN 模型**

对于 Faster RCNN 模型，我使用了 TensorFlow Object Detection 中的预训练模型。TensorFlow Object Detection 共享 COCO 预训练的 Faster RCNN，用于各种主干。使用了 Faster RCNN ResNet 50 主干。

### 测试结果

**yolo v5**

<div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\yolo-v5-1.gif" height="240" ></div><div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\yolo-v5-2.gif" height="240" ></div><div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\yolo-v5-3.gif" height="240" ></div>













**Faster-RCNN**

<div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\faster RCNN-1.gif" height="240" ></div><div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\faster RCNN-2.gif" height="240" ></div><div style="float:left;border:solid 1px 000;margin:2px;"><img src=".\img\faster RCNN-3.gif" height="240" ></div>













### 结论

- YOLO 模型似乎更善于检测较小的目标，在这种情况下是红绿灯，并且还能够在当汽车距离较远（即在透视上看起来较小）将其进行标记。YOLOv5s 的运行速度（端到端包括读取视频、运行模型和将结果保存到文件）为 52.8 FPS。而 Faser RCNN ResNet 50 的运行速度（端到端包括读取视频、运行模型和将结果保存到文件）为 21.7 FPS。
- Faster RCNN 模型在 60% 的阈值下运行，可以说它是用“Person”标签对人群进行标记，YOLO的结果干净整洁。不过，这两种模型在视频右下角的 abc（美国广播公司）徽标上都存在假正类误报。虽然运动球也是 COCO 的类别之一，但这两个模型都没有检测到篮球。**可能对快速或者加速运动的物体检测结果不够好**。
- 在最后一段视频中，我从 MOT 数据集中选择了一个室内拥挤的场景。这是一段很有挑战性的视频，因为光线不足，距离遥远，人群密集这一次的测试很有趣。当人们走进走廊的时候，这两种模型都很难检测到远处的人。这可能是由于光线较弱和目标较小所致。当人群靠近摄像机方向时，这两种模型都能对重叠的人进行标记

YOLOv5 在运行速度上有明显优势。小型 YOLOv5 模型运行速度加快了约 2.5 倍，同时在检测较小的目标时具有更好的性能。结果也更干净，几乎没有重叠的边框。Ultralytics 在他们的 YOLOv5 上做得非常出色，并开源了一个易于训练和运行推理的模型



















