# LargeScale_Hallucination_Detection

### Executive Summary

This repository encapsulates a scalable framework designed to detect hallucinations in large-scale data streams. By integrating machine learning models, streaming algorithms, and privacy-preserving techniques, the system ensures real-time analysis and decision-making in data-intensive environments.

### Repository Architecture
	•	ML-model: Contains trained machine learning model (logistics regression) and pkl files for hallucination detection.
	•	datasets: Includes datasets utilized for training and evaluation purposes.
	•	kafka-producer-consumer: Implements Apache Kafka producers and consumers for efficient data streaming.
	•	privacy-techniques: Houses methods ensuring data privacy and compliance during processing.
	•	streaming-algorithms: Features algorithms optimized for real-time data analysis.

### Features
	•	Streaming Data Pipeline with Kafka + Spark Structured Streaming
	•	ML Model Integration using logistic regression with TF-IDF features (Spark MLlib)
 	•	Entity Redaction for lightweight privacy preservation
  	•	Streaming Algorithms:
  		  •	 DGIM
  		  •	 Reservoir Sampling
		  •	 MinHash LSH
  	•	Dual Storage output to MongoDB and Snowflake

### Dataset
The FEVER (Fact Extraction and VERification) dataset is used as a proxy for hallucinated and factual claims. It includes over 185,000 labeled claims. Redacted claims are used as input to simulate anonymized, AI-generated statements.
