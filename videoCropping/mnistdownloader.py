from onnx import hub, save
m = hub.load("MNIST")   # downloads a vetted MNIST classifier
save(m, "mnist.onnx")
