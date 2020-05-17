from .model import build

def main():
    model = build(classes=200, learning_rate=0.00005, input_shape=(256, 256, 3))




if __name__ == '__main__':
    main()
