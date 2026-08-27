# Reparian_enchroachment_Detector
Traditional government remote-sensing models often rely on coarse or medium-resolution imagery. This leads to severe underestimations of informal structures inside protected riparian zones because clustered homes blend together into a single pixel grid (e.g., detecting only 118 structures when 700 actually exist).
The goal is to accurately isolate and approximate the real-world count of 700 structures within a 60-meter riparian buffer zone.

# Business Understanding
## 1.1 Problem Statement
Riparian zones in rapidly urbanizing areas are increasingly affected by informal settlements, land-use change, and encroachment. Accurate detection and monitoring of structures within protected riparian buffers is essential for environmental management, urban planning, flood-risk mitigation, and regulatory enforcement.

Traditional remote-sensing methods often rely on medium- or coarse-resolution imagery and pixel-based classification, which may fail to distinguish small, closely spaced structures in densely populated areas. This can lead to significant underestimation—for example, detecting approximately 118 structures where community assessments indicate about 700.

Manual ground surveys provide greater detail but are costly, time-consuming, difficult to scale, and potentially unsafe. There is therefore a need for a scalable spatial-analytics approach that combines high-resolution remote sensing with object-level structure detection to improve the identification and counting of riparian encroachment.

## 1.2 Main Goal

To provide an accurate, scalable way of identifying and counting individual buildings encroaching on Nairobi's riparian buffer zones without relying entirely on physical ground surveys

## 1.3  Main Objective
To develop a high-resolution machine-learning and GIS-based pipeline for detecting, mapping, and counting individual buildings within the 60-meter riparian buffer in Kasarani, Nairobi.

## Project Scope
This project will focus on developing a high-resolution machine learning and GIS pipeline to detect, map, and count individual buildings within a 60-meter riparian buffer in Kasarani, Nairobi.

The project will:

- Use high-resolution satellite or aerial imagery and supporting GIS datasets.
- Develop a Random Forest baseline for pixel-level building/roof classification.
- Develop a deep-learning model using transfer learning to detect individual buildings, including closely packed structures.
- Use GIS to identify structures located within the 60-meter riparian buffer.
- Compare model performance using appropriate detection and counting metrics.
- Produce maps and structure counts that can support riparian monitoring and planning.

## Success Criteria
The project will be considered successful if it can:

- Accurately detect individual buildings within the 60-meter riparian buffer, including closely packed structures.
- Reduce building undercounting and merging compared with the Random Forest pixel-level baseline.
- Achieve strong precision, recall, F1-score, and IoU on the test dataset.
- Produce a building count that reasonably approximates the available ground-truth or manual survey data.
- Generate clear GIS maps showing the location and distribution of structures within the riparian zone.
- Demonstrate that the resulting workflow is reproducible and scalable to other riparian areas.
