Growlytics – Smart Agriculture Using AI/ML


Problem Definition

Productivity needs to be increased so that farmers can get more pay from the same piece of land without degrading the soil.  
Indian farmers often face difficulty in selecting the **right crop** based on their soil requirements depending upon factors such as **Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, rainfall, and pH**.

Farmers are also generally unaware about the **type of fertilizer (organic or standard)** that suits their soil condition.  
Due to **inadequate and imbalanced fertilization**, soil degradation is occurring, leading to **nutrient mining** and **second-generation problems in nutrient management**.

According to a study by the **Associated Chambers of Commerce and Industry of India (ASSOCHAM)**, annual crop losses due to pests amount to nearly **₹50,000 crore**.


Objective

1. Implement **Precision Agriculture** – a modern farming technique that uses soil characteristics, soil type, and crop yield data to recommend the right crop for a specific location.  
2. Propose a **Crop Recommendation System** using an **Ensemble Model** with **Majority Voting Technique** for high accuracy and efficiency.  
3. Recommend **Fertilizers** based on **N, P, K** values and the chosen crop.  
4. Recognize **Pests** using Deep Learning and recommend appropriate **Pesticides** available in India (as per ISO standards).


Methodology

Crop Recommendation
**Steps:**
1. **Dataset Acquisition** – Data on soil nutrients, rainfall, temperature, humidity, and crop yield.  
2. **Input Parameters:** N, P, K, temperature, humidity, rainfall, and pH.  
3. **Model Training:** ML models used –  
   - Support Vector Machine (SVM)  
   - Random Forest  
   - Naive Bayes  
   - K-Nearest Neighbors (KNN)  
4. **Technique:** Ensemble Learning with Majority Voting for improved accuracy.  
5. **Output:** Recommended crop for given input data.  
6. **Model Storage:** `.pkl` model file generated for future use.



Fertilizer Recommendation
**Steps:**
1. **Dataset Acquisition** – Fertilizer data from credible sources:  
   - *The Fertilizer Association of India (FAI)*  
   - *Indian Institute of Water Management (IIWM)*  
2. **Input Parameters:** Crop type and actual N, P, K values.  
3. **Comparison:**  
   - High → Excess nutrient  
   - Low → Deficient nutrient  
   - Up to the mark → Balanced  
4. **Approach:**  
   - Use **dictionary-based suggestions** from verified agricultural sources.  
5. **Output:** Suggested fertilizer with nutrient balancing tips.



Pesticide Recommendation
**Steps:**
1. **Data Acquisition:** Image scraping from Google Images for pest datasets.  
2. **Labeling:** Each image labeled with pest type.  
3. **Data Cleaning & Augmentation:**  
   - Remove irrelevant data.  
   - Augment images to increase dataset diversity.  
4. **Model Creation:** Deep Learning (CNN) model trained for pest detection.  
5. **Output:**  
   - Detected pest type.  
   - Recommended pesticide as per ISO standards.



Tools & Technologies
| Category | Tools Used |
|-----------|-------------|
| **Programming Language** | Python |
| **Frameworks** | Flask, TensorFlow/Keras |
| **ML Libraries** | scikit-learn, pandas, numpy |
| **Dataset Handling** | CSV, pickle (.pkl) |
| **Hardware (optional IoT)** | Arduino, DHT11, Soil Moisture Sensor, Relay Module |



Conclusion

India’s farmers are hard at work, feeding a nation of nearly **1.4 billion** people.  
However, their productivity is often threatened by **natural factors** such as poor soil management, nutrient imbalance, and pest attacks.

This project provides a **comprehensive precision agriculture solution** that:
- Helps farmers **maximize productivity**  
- Reduces **soil degradation**  
- Offers **fertilizer recommendations** based on soil health  
- Suggests **appropriate crops** for sustainable yield  
- Detects **pests** and recommends **suitable pesticides**

In essence, **Growlytics** supports farmers in making **data-driven agricultural decisions** — benefiting both **farmers and the environment**.

Run Locally
```bash
git clone https://github.com/NipunShanbhag/Growlytics.git
cd Growlytics
pip install -r requirements.txt
python app.py


Author
**Nipun Naresh Shanbhag**  
[shanbhagnipun2003@gmail.com](mailto:shanbhagnipun2003@gmail.com) 
[GitHub Profile](https://github.com/NipunShanbhag)
