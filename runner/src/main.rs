use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use tch::{nn, nn::Module, Device, vision};

#[derive(Debug)]
struct BlurDetectionModel {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl BlurDetectionModel {
    fn new(vs: &nn::Path) -> BlurDetectionModel {
        let conv1 = nn::conv2d(vs, 3, 64, 3, Default::default());
        let conv2 = nn::conv2d(vs, 64, 128, 3, Default::default());
        let fc1 = nn::linear(vs / "fc1", 128 * 64 * 64, 256, Default::default());
        let fc2 = nn::linear(vs / "fc2", 256, 1, Default::default());

        BlurDetectionModel {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::Module for BlurDetectionModel {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let xs = xs.apply(&self.conv1).relu().max_pool2d_default(2);
        let xs = xs.apply(&self.conv2).relu().max_pool2d_default(2);
        let xs = xs.view((-1, 128 * 64 * 64));
        let xs = xs.apply(&self.fc1).relu();
        xs.apply(&self.fc2).sigmoid()
    }
}

#[derive(Serialize, Deserialize)]
struct Prediction {
    status: String,
    filename: String,
    detected: Detected,
}

#[derive(Serialize, Deserialize)]
struct Detected {
    blurry: f64,
}

fn main() {
    let device = Device::cuda_if_available();
    let mut vs = tch::nn::VarStore::new(device);
    let model = BlurDetectionModel::new(&vs.root());
    vs.load("../blur_detection_model.tch").unwrap();

    // Make predictions on new data
    let image_dir = Path::new("./images");
    let mut predictions = Vec::<Prediction>::new();

    for entry in fs::read_dir(image_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            match vision::image::load(&path) {
                Ok(img) => {
                    let img = img.to_device(device);
                    let output = model.forward(&img.unsqueeze(0)).double_value(&[]);
                    predictions.push(Prediction {
                        status: "success".to_string(),
                        filename: path.to_str().unwrap().to_string(),
                        detected: Detected { blurry: output },
                    });
                }
                Err(_) => {
                    predictions.push(Prediction {
                        status: "failure".to_string(),
                        filename: path.to_str().unwrap().to_string(),
                        detected: Detected { blurry: 0.0 },
                    });
                }
            }
        }
    }

    // Output the predictions as a JSON
    println!("{}", serde_json::to_string_pretty(&predictions).unwrap());
}
