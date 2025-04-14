// This file is part of the open-source port of SeetaFace engine, which originally includes three modules:
//      SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
//
// This file is part of the SeetaFace Detection module, containing codes implementing the face detection method described in the following paper:
//
//      Funnel-structured cascade for multi-view face detection with alignment awareness,
//      Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
//      In Neurocomputing (under review)
//
// Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
// Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
//
// As an open-source face recognition engine: you can redistribute SeetaFace source codes
// and/or modify it under the terms of the BSD 2-Clause License.
//
// You should have received a copy of the BSD 2-Clause License along with the software.
// If not, see < https://opensource.org/licenses/BSD-2-Clause>.

use std::env::Args;
use std::time::{Duration, Instant};
use std::fs;
use std::path::Path;
use image::{GenericImageView, ImageBuffer,};

use image::{DynamicImage, GrayImage, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use rustface::{Detector, FaceInfo, ImageData};
use walkdir::WalkDir;
use rayon::prelude::*;


const OUTPUT_FILE: &str = "test.png";

fn main() {

    // save the start time
    let start_time = Instant::now();

    // let input_dir = "WIDER_val/WIDER_val/images";

    // Create output directory(To store the complete processed image with detected faces) if it doesn't exist
    // let output_dir = "output_dir";
    // fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // New directory to store cropped face boxes
    let boxes_dir = "cropped_faces";
    fs::create_dir_all(boxes_dir).expect("Failed to create boxes directory");

    let options = match Options::parse(std::env::args()) {
        Ok(options) => options,
        Err(message) => {
            println!("Failed to parse program arguments: {}", message);
            std::process::exit(1)
        }
    };

    let mut detector = match rustface::create_detector(options.model_path()) {
        Ok(detector) => detector,
        Err(error) => {
            println!("Failed to create detector: {}", error);
            std::process::exit(1)
        }
    };

    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);

    let entries: Vec<_> = WalkDir::new(options.image_dir_path())
    .into_iter()
    .filter_map(Result::ok)
    .filter(|e| e.path().is_file())
    .collect();

    entries.par_iter().for_each(|entry| 
    {
            let path = entry.path();
            match image::open(path)
            {
            // let image: DynamicImage = match image::open(options.image_path()) {
                Ok(image) => {
                    println!("Processing image: ");

                    let mut detector = match rustface::create_detector(options.model_path()) {
                        Ok(detector) => detector,
                        Err(error) => {
                            println!("Failed to create detector: {}", error);
                            std::process::exit(1)
                        }
                    };

                    let mut rgb = image.to_rgb8();
                    let faces = detect_faces(&mut *detector, &image.to_luma8());

                    for face in &faces {
                        let bbox = face.bbox();
                        let rect = Rect::at(bbox.x(), bbox.y()).of_size(bbox.width(), bbox.height());
                
                        draw_hollow_rect_mut(&mut rgb, rect, Rgb([255, 0, 0]));
                    }

                    // Add a counter to save multiple faces from the same image
                    let mut face_index = 0;

                    for face in &faces {
                        let bbox = face.bbox();
                        let x = bbox.x().max(0);
                        let y = bbox.y().max(0);
                        let w = bbox.width().min(rgb.width() - (x as u32));
                        let h = bbox.height().min(rgb.height() - (y as u32));

                        // Crop the face region
                        let face_crop = image.crop_imm(x as u32, y as u32, w, h);
                        
                        // Save it with a unique filename
                        let file_stem = path.file_stem().unwrap().to_str().unwrap();
                        let face_filename = format!("{}_{}.png", file_stem, face_index);
                        let face_output_path = Path::new(boxes_dir).join(&face_filename);

                        if let Err(e) = face_crop.save(face_output_path) {
                            eprintln!("Failed to save face crop {}: {}", face_filename, e);
                        }

                        face_index += 1;
                    }

                    // // To save the complete processed image with detected faces
                    // let file_name = path.file_name().unwrap().to_str().unwrap();
                    // let output_path = Path::new(output_dir).join(file_name);
                    // if let Err(e) = rgb.save(output_path) {
                    //     eprintln!("Failed to save image {}: {}", file_name, e);
                    // } else {
                    //     println!("Processed and saved: {}", file_name);
                    // }

                }
                Err(message) => {
                    println!("Failed to read image: {}", message);
                    std::process::exit(1)
                    }
                };
            });
    // Calculate the elapsed time
    let elapsed_time = start_time.elapsed();
    println!("Total elapsed time: {} seconds", elapsed_time.as_secs());
    }

fn detect_faces(detector: &mut dyn Detector, gray: &GrayImage) -> Vec<FaceInfo> {
    let (width, height) = gray.dimensions();
    let image = ImageData::new(gray, width, height);
    let now = Instant::now();
    let faces = detector.detect(&image);
    println!(
        "Found {} faces in {} ms",
        faces.len(),
        get_millis(now.elapsed())
    );
    faces
}

fn get_millis(duration: Duration) -> u64 {
    duration.as_secs() * 1000u64 + u64::from(duration.subsec_millis())
}

struct Options {
    image_dir_path: String,
    model_path: String,
}

impl Options {
    fn parse(args: Args) -> Result<Self, String> {
        let args: Vec<String> = args.into_iter().collect();
        if args.len() != 3 {
            return Err(format!("Usage: {} <model-path> <image-path>", args[0]));
        }

        let model_path = args[1].clone();
        let image_dir_path = args[2].clone();

        Ok(Options {
            image_dir_path,
            model_path,
        })
    }

    fn image_dir_path(&self) -> &str {
        &self.image_dir_path[..]
    }

    fn model_path(&self) -> &str {
        &self.model_path[..]
    }
}
