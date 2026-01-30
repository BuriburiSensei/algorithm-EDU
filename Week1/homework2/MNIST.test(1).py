import numpy as np
import gzip
import struct



np.random.seed(42)
input_dim = 784
output_dim = 10
hidden_dim = 128
learning_rate = 0.5
batch_size = 128
epochs = 50
val_ratio = 1/6

def load_mnist(image_path, label_path):
    with gzip.open(image_path, 'rb') as f:
        magic,num,rows,cols = struct.unpack('>4I',f.read(16))
        images = np.frombuffer(f.read(),dtype=np.uint8).reshape(num, rows * cols)
        images = images.astype(np.float32)/255.0
    with gzip.open(label_path, 'rb') as f:
        magic,num = struct.unpack('>2I',f.read(8))
        labels = np.frombuffer(f.read(),dtype=np.uint8)
    return images, labels

def one_hot_encode(labels,num_classes):
    return np.eye(num_classes)[labels]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return (x) * (1 - (x))

def softmax(x):
    exp_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def init_network():
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / (input_dim + hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / (hidden_dim + output_dim))
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

def forward_propagation(x,W1, b1, W2, b2):
    z1=x @ W1 + b1
    h1=sigmoid(z1)
    z2=h1 @ W2 + b2
    y_pred=softmax(z2)
    return h1, y_pred

def backward_propagation(x, y_true, h1, y_pred, W2):
    batch_size = x.shape[0]
    dz2=y_pred - y_true
    dW2=(h1.T@dz2)/batch_size
    db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size
    dz1 = dz2 @ W2.T * sigmoid_deriv(h1)
    dW1 = (x.T @ dz1) / batch_size
    db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
    return dW1, db1, dW2, db2

def cross_entropy_loss(y_true,y_pred):
    epsilon = 1e-8
    return -np.mean(np.sum(y_true*np.log(y_pred+epsilon),axis=1))

def evaluate(x, y_onehot, W1, b1, W2, b2):
    _, y_pred = forward_propagation(x, W1, b1, W2, b2)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_onehot, axis=1)
    return np.mean(y_pred_label == y_true_label)

def predict(x, W1, b1, W2, b2):
    _, y_pred = forward_propagation(x, W1, b1, W2, b2)
    return np.argmax(y_pred, axis=1)

def train_model(x_train, y_train, x_val, y_val, W1, b1, W2, b2):
    n_samples = x_train.shape[0]
    val_accuracies = []
    for epoch in range(1, epochs + 1):
        shuffle_idx = np.random.permutation(n_samples)
        x_train_shuffle = x_train[shuffle_idx]
        y_train_shuffle = y_train[shuffle_idx]
        total_loss = 0.0
        n_batches = n_samples // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_train_shuffle[start:end]
            y_batch = y_train_shuffle[start:end]
            h1_batch, y_pred_batch = forward_propagation(x_batch, W1, b1, W2, b2)
            loss_batch = cross_entropy_loss(y_batch, y_pred_batch)
            total_loss += loss_batch
            dW1, db1, dW2, db2 = backward_propagation(x_batch, y_batch, h1_batch, y_pred_batch, W2)
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        avg_loss = total_loss / n_batches
        train_acc = evaluate(x_train, y_train, W1, b1, W2, b2)
        val_acc = evaluate(x_val, y_val, W1, b1, W2, b2)
        val_accuracies.append(val_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}/{epochs} | 平均损失: {avg_loss:.4f} | 训练准确率: {train_acc:.4f} | 验证准确率: {val_acc:.4f}")

    return W1, b1, W2, b2, val_accuracies

def main():
    print("===== 开始加载本地MNIST数据集 =====")
    x_train_raw, y_train_raw = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    x_test_raw, y_test_raw = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    print(f"数据集加载完成 | 训练集: {x_train_raw.shape} | 测试集: {x_test_raw.shape}")

    print("===== 拆分训练集/验证集 + One-Hot编码 =====")
    val_size = int(len(x_train_raw) * val_ratio)
    x_train, x_val = x_train_raw[val_size:], x_train_raw[:val_size]
    y_train, y_val = y_train_raw[val_size:], y_train_raw[:val_size]
    y_train_onehot = one_hot_encode(y_train, output_dim)
    y_val_onehot = one_hot_encode(y_val, output_dim)
    y_test_onehot = one_hot_encode(y_test_raw, output_dim)
    print(f"数据集拆分完成 | 训练集: {x_train.shape} | 验证集: {x_val.shape} | 测试集: {x_test_raw.shape}")

    print("===== 初始化2层神经网络参数 =====")
    W1, b1, W2, b2 = init_network()

    print("===== 开始训练模型（50轮，批次64） =====")
    W1_trained, b1_trained, W2_trained, b2_trained, val_accs = train_model(
        x_train, y_train_onehot, x_val, y_val_onehot, W1, b1, W2, b2
    )

    print("===== 测试集最终评估 =====")
    test_acc = evaluate(x_test_raw, y_test_onehot, W1_trained, b1_trained, W2_trained, b2_trained)
    print(f"✅ 训练完成！测试集最终准确率: {test_acc:.4f}")

    print("===== 预测示例（测试集前5张图片） =====")
    for i in range(5):
        img = x_test_raw[i:i+1]
        true_label = y_test_raw[i]
        pred_label = predict(img, W1_trained, b1_trained, W2_trained, b2_trained)[0]
        print(f"第{i + 1}张 | 真实标签: {true_label} | 预测标签: {pred_label}")

if __name__ == "__main__":
    main()