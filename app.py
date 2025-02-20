from joblib import load
from flask import Flask, request, jsonify, render_template
from model_helper import load_tflite_model,predict_image_tflite,load_model_and_preprocessors,get_top_recommendations,predict_with_model,load_nlp_models,find_relevant_text,get_embedding,load_segmentation_model,create_cluster_only_graph,predict_image
import numpy as np
import traceback
import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np
import traceback
import plotly.graph_objects as go
import io
from PIL import Image
# Load the saved model

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calories')
def calories_page():
    return render_template('calories.html')

@app.route('/diet')
def diet_page():
    return render_template('diet.html')

@app.route('/exercise')
def exercise_page():
    return render_template('exercise.html')

@app.route('/predict_calories', methods=['POST'])
def predict_calories():
    try:
        data = request.json
        gender = data['gender']  # 0 or 1
        heartrate = data['heartrate']
        body_temp = data['body_temp']
        height = data['height']
        age = data['age']
        
        inputs = [[float(gender), float(age), float(height), float(heartrate), float(body_temp)]]
        model = load("models/calories/xgr_calories.joblib")

        calories = model.predict(inputs)[0]
        
        return jsonify({'calories': float(calories)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_diet', methods=['POST'])
def predict_diet():
    try:
        data = request.json
        inputs =np.asarray( [[data['Age'], data['Height'], data['Weight'], data['BMI'],
                  data['Sex'], data['Level'], data['Fitness Goal'], data['Fitness Type'],
                  data['Hypertension'], data['Diabetes']]])
        path="models/gym_diet"
        model_name="diet"
        model, mlb, scaler,label_dict=load_model_and_preprocessors(path,model_name)
        diet_recommendations = predict_with_model(model, scaler, mlb, inputs, top_n=3,label_encoder=label_dict)
        final=diet_recommendations[0]
        out=""
        for i in final:
            out=out+"    "+i
        
        
        return jsonify({'diet_recommendation': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_exercise', methods=['POST'])
def predict_exercise():
    try:
        data = request.json
        inputs = np.asarray([[data['Age'], data['Height'], data['Weight'], data['BMI'],
                  data['Sex'], data['Level'], data['Fitness Goal'], data['Fitness Type'],
                  data['Hypertension'], data['Diabetes']]])
        print(inputs.shape)
        path="models/gym_diet"
        model_name="exercise"
        model, mlb, scaler,label_dict=load_model_and_preprocessors(path,model_name)
        exer_recommendations = predict_with_model(model, scaler, mlb, inputs, top_n=3,label_encoder=label_dict)
        final=exer_recommendations[0]
        out=""
        for i in final:
            out=out+"    "+i
        print("Diet recommendations for sample inputs:", out)
        
        
        return jsonify({'exercise_recommendation': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/segmentation')
def segmentation_page():
    """Renders segmentation.html with the default cluster-only graph."""
    graph_html = create_cluster_only_graph()
    return render_template('segmentation.html', graph_html=graph_html)

@app.route('/predict_segment', methods=['POST'])
def predict_segment():
    try:
        data = request.json
        kmeans_model,scaler, df_grouped=load_segmentation_model()

        # Extract user inputs and create DataFrame
        columns = ['TotalSteps', 'LoggedActivitiesDistance', 'ModeratelyActiveDistance',
                   'LightActiveDistance', 'SedentaryActiveDistance', 'SedentaryMinutes', 'Calories']
        user_inputs_df = pd.DataFrame([[data[col] for col in columns]], columns=columns)

        # Scale user inputs
        user_scaled = scaler.transform(user_inputs_df)

        # Predict cluster
        cluster_label = kmeans_model.predict(user_scaled)[0]

        # Convert distances from miles to meters for display
        distance_columns = ["LoggedActivitiesDistance", "ModeratelyActiveDistance",
                            "LightActiveDistance", "SedentaryActiveDistance"]
        user_inputs_df[distance_columns] *= 1609.34  

        # Create radar chart (Clusters + User Data)
        categories = df_grouped.columns.tolist()
        fig = go.Figure()

        # Plot clusters
        for cluster in df_grouped.index:
            fig.add_trace(go.Scatterpolar(
                r=df_grouped.loc[cluster].values,  
                theta=categories,  
                fill='toself',
                name=f'Cluster {cluster}',
                opacity=0.5
            ))

        # Plot user input
        fig.add_trace(go.Scatterpolar(
            r=user_inputs_df.iloc[0].values,  # Use DataFrame values directly
            theta=categories,  
            fill='none',  
            name="User Progress",
            line=dict(color='black', width=3, dash='solid'),
            marker=dict(color='black', size=6)
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, gridcolor="lightgrey")),
            title="User Segmentation - Cluster Comparison",
            showlegend=True,
            font=dict(size=14)
        )

        

        # Convert to HTML
        graph_html = fig.to_html(full_html=False)
        if cluster_label==0:
            cluster_label="Light Mover"
        else:
            cluster_label="Pro Mover"
        # Return JSON response (not render_template)
        #return jsonify({'graph_html': graph_html, 'cluster_name': cluster_label})
        return render_template('segementation_result.html', graph_html=graph_html, cluster_name=cluster_label)

    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return jsonify({'error': error_message, 'traceback': error_traceback}), 400

@app.route('/FactCheck')
def factcheck_page():
    return render_template('factcheck.html')

@app.route('/fact_check', methods=['POST'])
def fact_check():
    try:
        data = request.json
        user_question = data['question']
        model, tokenizer,chunk_embeddings,text_chunks=load_nlp_models()
        
        best_match, score = find_relevant_text(user_question,text_chunks,chunk_embeddings,get_embedding,model,tokenizer)
        print(score)
        print(best_match)

        return jsonify({'best_match': best_match, 'score': round(float(score), 2)})
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return jsonify({'error': error_message, 'traceback': error_traceback}), 400
@app.route('/ExerciseClassification')
def exercise_classification_page():
    return render_template('exerciseclassification.html')
@app.route('/predict_exercise_class', methods=['POST'])
def predict_exercise_class():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        image_stream = io.BytesIO(image.read())
        pil_image = Image.open(image_stream).convert("RGB")  # Ensure RGB format
        pil_image.save("temp.png")
        interpreter, class_labels=load_tflite_model(tflite_path="models/image_classifer/mobilenetv2_posture_quantized.tflite",class_labels_path="models/image_classifer/class_labels.npy")
        
        prediction = predict_image_tflite(class_labels=class_labels,interpreter=interpreter,image_path="temp.png")
        
        return jsonify({'class': str(prediction)})
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return jsonify({'error': error_message, 'traceback': error_traceback}), 400
@app.route('/Yogaclassification')
def yoga_classification_page():
    return render_template('yogaclassification.html')

@app.route('/predict_yoga_class', methods=['POST'])
def predict_yoga_class():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        image_stream = io.BytesIO(image.read())
        pil_image = Image.open(image_stream).convert("RGB")  # Ensure RGB format
        pil_image.save("temp.png")
        interpreter, class_labels=load_tflite_model()
        
        prediction = predict_image_tflite(class_labels=class_labels,interpreter=interpreter,image_path="temp.png")
        
        return jsonify({'class': str(prediction)})
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return jsonify({'error': error_message, 'traceback': error_traceback}), 400

if __name__ == '__main__':
    app.run(debug=True)
