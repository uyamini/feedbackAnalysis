import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 1: Fetch and process the data
def fetch_and_preprocess_data():
    feedback_df = fetch_feedback_data()
    feedback_df['cleaned_text'] = feedback_df['feedback_text'].apply(preprocess_text)
    return feedback_df

# Step 2: Load models and evaluate
def evaluate_model(model_name, feedback_df):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Tokenize data
    def preprocess_data(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    dataset = load_dataset('csv', data_files={'data': feedback_df.to_csv(index=False)})['data']
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    
    # Training arguments (no training, just evaluation)
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
        logging_dir='./logs',
    )
    
    # Initialize Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )
    
    # Evaluate
    eval_results = trainer.evaluate()
    return eval_results

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Step 3: Compare models and plot results
def compare_models(feedback_df):
    models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
    results = {}
    
    for model_name in models:
        results[model_name] = evaluate_model(model_name, feedback_df)
    
    return results

# Fetch, preprocess data, and compare models
feedback_df = fetch_and_preprocess_data()
model_results = compare_models(feedback_df)

# Step 4: Create a Dash application
app = dash.Dash(__name__)

# Step 5: Create the layout of the app
app.layout = html.Div([
    html.H1("Model Performance Comparison", style={'text-align': 'center'}),
    
    # Graph component for model comparison
    dcc.Graph(id='model-comparison-graph'),
    
    # Dropdown to select metric to display
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Accuracy', 'value': 'accuracy'},
            {'label': 'Precision', 'value': 'precision'},
            {'label': 'Recall', 'value': 'recall'},
            {'label': 'F1 Score', 'value': 'f1'}
        ],
        value='accuracy',
        clearable=False
    ),
])

# Step 6: Define the callbacks for interactivity
@app.callback(
    Output('model-comparison-graph', 'figure'),
    Input('metric-dropdown', 'value')
)
def update_graph(selected_metric):
    metrics = {model: model_results[model][selected_metric] for model in model_results.keys()}
    
    # Create the bar chart using Plotly Graph Objects
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        text=list(metrics.values()),
        textposition='auto',
        marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c'])  # blue, orange, green for different models
    ))

    # Update layout to improve appearance
    fig.update_layout(
        title=f'Model Comparison: {selected_metric.capitalize()}',
        xaxis_title='Model',
        yaxis_title=selected_metric.capitalize(),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="black"),
        title_x=0.5
    )

    return fig

# Step 7: Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
