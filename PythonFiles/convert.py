import tensorflow as tf
import tf2onnx
import onnx


input_signature = [tf.TensorSpec([None, 128, 128, 1], tf.float32, name="input")]
model = tf.keras.models.load_model(r"PythonFiles\best_model6.keras")
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "model6.onnx")
print("ONNX model saved as model6.onnx")