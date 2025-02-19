import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model  # For loading Keras models
from transformers import AutoTokenizer, TFAutoModel
from numpy.linalg import norm
from joblib import load
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
import pickle
def load_model_and_preprocessors(path,model_name):
    preprocessor_path=path+"/preprocessor/"
    model_path=path+"/neural_nets/{}_model.h5".format(model_name)
    model = load_model(model_path)
    scaler=joblib.load(preprocessor_path+"scaler.pkl")
    with open(preprocessor_path+"model_paths.json", "r") as file:
        label_dict = json.load(file) 
    for name,file_name in label_dict.items():
        label_dict[name]=joblib.load(preprocessor_path+file_name)
    mlb=joblib.load(preprocessor_path+"mlb_"+model_name+".pkl")

    #mlb = joblib.load(model_path + "_mlb.pkl")
    #scaler = joblib.load(model_path + "_scaler.pkl")
    return model, mlb, scaler,label_dict




def get_top_recommendations(predictions, mlb, top_n=3):
    recommended_items = []
    for pred in predictions:
        top_indices = np.argsort(pred)[-top_n:][::-1]  # Get indices of top N values
        top_items = [mlb.classes_[i] for i in top_indices if pred[i] > 0.5]  # Include only if above threshold
        recommended_items.append(top_items)
    return recommended_items
def predict_with_model(model, scaler, mlb, input_data,label_encoder, top_n=3):
    scaler_data=np.asarray([input_data[0][:4]])
    input_data_scaled = scaler.transform(scaler_data)
    label_data=input_data[0][4:]
    label_data_encoded=[]
    count=0
    for key,value in label_encoder.items():
        data_point=np.asarray([label_data[count]])
        label_data_encoded.append(value.transform(data_point))
        count=count+1
    label_data_encoded=[i[0] for i in label_data_encoded]
    ids=list(input_data_scaled[0])
    ids.extend(label_data_encoded)
    final_data=np.asarray(ids)
    final_data=final_data.reshape(1,10)
    print("****")
    print(final_data)
    predictions = model.predict(final_data)
    return get_top_recommendations(predictions, mlb, top_n)
def load_segmentation_model():
    # Load KMeans model and scaler
    kmeans_model = load("models/user_segmentation/kmeans.pkl")
    scaler = load("models/user_segmentation/scaler.pkl")
    # Load cluster centroids
    df_grouped = pd.read_csv("models/user_segmentation/cluster_centroid.csv", index_col=0)
    # Scale TotalSteps and Calories for display
    df_grouped["TotalSteps"] /= 1
    df_grouped["Calories"] /= 1
    # Convert miles to meters
    for col in ["LoggedActivitiesDistance", "ModeratelyActiveDistance", 
            "LightActiveDistance", "SedentaryActiveDistance"]:
        df_grouped[col] *= 1609.34
    return kmeans_model,scaler, df_grouped
def create_cluster_only_graph():
    """Generates a radar chart with only cluster means (default view)."""
    kmeans_model,scaler, df_grouped=load_segmentation_model()

    categories = df_grouped.columns.tolist()
    fig = go.Figure()

    for cluster in df_grouped.index:
        fig.add_trace(go.Scatterpolar(
            r=df_grouped.loc[cluster].values,  
            theta=categories,  
            fill='toself',
            name=f'Cluster {cluster}',
            opacity=0.5
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor="lightgrey")),
        title="User Segmentation - Cluster Comparison",
        showlegend=True,
        font=dict(size=14)
    )

    return fig.to_html(full_html=False)


def load_nlp_models():
    # Load model and tokenizer
    model = TFAutoModel.from_pretrained("models/nlp/bert_model")
    tokenizer = AutoTokenizer.from_pretrained("models/nlp/bert_tokenizer")
    chunk_embeddings = np.load("models/nlp/embeddings.npy")
    text_chunks = np.load("models/nlp/text_chunks.npy")

    return model, tokenizer,chunk_embeddings,text_chunks

def find_relevant_text(user_question, text_chunks, chunk_embeddings, get_embedding,model,tokenizer):
    # Convert user question into an embedding
    question_embedding = get_embedding(user_question,model,tokenizer).flatten()

    # Compute cosine similarity
    similarities = np.dot(chunk_embeddings, question_embedding)

    # Ensure similarity shape matches the number of text chunks
    assert similarities.shape[0] == len(text_chunks), "Mismatch in similarity scores and text chunks!"

    # Normalize (avoid division by zero)
    chunk_norms = norm(chunk_embeddings, axis=1)
    question_norm = norm(question_embedding)
    similarities = similarities / (chunk_norms * question_norm + 1e-8)



    # Get the best matching chunk
    best_match_index = np.argmax(similarities)

    # Ensure index is within bounds
    if best_match_index >= len(text_chunks):
        raise IndexError(f"Index {best_match_index} is out of bounds! Check text chunk and embedding sizes.")

    best_match_score = similarities[best_match_index]
    return text_chunks[best_match_index], best_match_score
def get_embedding(text,model,tokenizer):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    outputs = model(**inputs)
    return np.mean(outputs.last_hidden_state.numpy(), axis=1) 
def predict_image(image_path, model_path="models/image_classifer/cnn_posture_classifier.h5", preprocessor_path="models/image_classifer/preprocessor.pkl", class_labels_path="models/image_classifer/categories.npy"):
    model = tf.keras.models.load_model(model_path)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocessor.standardize(img_array)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels=np.load(class_labels_path)
    print(class_labels)
    print(predicted_class)
    class_name = class_labels[predicted_class] 
    
    print(f"Predicted class: {class_name}")
    return class_name

def load_tflite_model(tflite_path="models/yoga/mobilenetv2_yoga_quantized.tflite", class_labels_path="models/yoga/class_labels.npy"):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    class_labels = np.load(class_labels_path, allow_pickle=True)
    return interpreter, class_labels.tolist()


def predict_image_tflite(interpreter, image_path, class_labels, image_size=(128, 128)):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimensionk
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class]
