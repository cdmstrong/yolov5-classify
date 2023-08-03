from roboflow import Roboflow
rf = Roboflow(api_key="CQ0Pr9ov4LjzPofsH4K9")
project = rf.workspace("").project("-car-heavy-truck-tanker-excavator-non-motorized")
dataset = project.version(1).download("yolov5")