use tch::{nn, nn::Module, nn::VarStore, Device, Tensor};
use image::DynamicImage;

#[derive(Debug)]
struct BlurClassifier {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl BlurClassifier {
    fn new(vs: &nn::Path) -> BlurClassifier {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let linear1 = nn::linear(vs, 1024, 128, Default::default());
        let linear2 = nn::linear(vs, 128, 2, Default::default());

        BlurClassifier {
            conv1,
            conv2,
            linear1,
            linear2,
        }
    }
}

impl nn::Module for BlurClassifier {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 1, 256, 256]) // Adjust dimensions based on your input size
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .relu()
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .relu()
            .view([-1, 1024])
            .apply(&self.linear1)
            .relu()
            .apply(&self.linear2)
    }
}


fn preprocess_image(image: &DynamicImage) -> Tensor {
    let resized_image = image.resize(256, 256, image::imageops::Nearest).to_luma8();

    let tensor_data: Vec<f32> = resized_image
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect();

Tensor::of_slice(&tensor_data)
}

fn main() {
    let image_path = "path/to/your/image.jpg"; // Replace with your image path

    let mut vs = VarStore::new(Device::Cpu);
    let net = BlurClassifier::new(&vs.root());

    // Load a pre-trained model (adjust the path accordingly)
    vs.load("path/to/your/pretrained_model.tch")
        .expect("Failed to load pre-trained model.");

    match image::open(image_path) {
        Ok(image) => {
            // Preprocess the image and convert to tensor
            let image_tensor = preprocess_image(&image);

            // Run the image through the model
            let output = net.forward(&image_tensor);

            // Get the class probabilities (0 = not blurry, 1 = blurry)
            let probabilities = output.softmax(-1, tch::Kind::Float);

            let blurry_prob = probabilities.double_value(&[0]);
            let not_blurry_prob = probabilities.double_value(&[1]);

            println!("Blurry Probability: {}", blurry_prob);
            println!("Not Blurry Probability: {}", not_blurry_prob);

            if blurry_prob > not_blurry_prob {
                println!("The image is blurry.");
            } else {
                println!("The image is not blurry.");
            }
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }
}