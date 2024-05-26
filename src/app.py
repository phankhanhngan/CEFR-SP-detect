from flask_cors import CORS

import torch
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
from model import LevelEstimaterClassification

app = Flask(__name__)


def map_to_cefr(score: float) -> str:
    """Map a numeric score to a CEFR level."""
    if 0 <= score < 1:
        return 'A1'
    elif 1 <= score < 2:
        return 'A2'
    elif 2 <= score < 3:
        return 'B1'
    elif 3 <= score < 4:
        return 'B2'
    elif 4 <= score < 5:
        return 'C1'
    else:
        return 'C2'


# Load model and tokenizer
model_path = './cefr-sp/level_estimator-epoch=10-val_score=0.797473.ckpt'
encoder = 'bert-base-cased'
model_type = 'regression'  # or 'metric'

model_class = {
    'regression': LevelEstimaterClassification
}[model_type]
tokenizer = AutoTokenizer.from_pretrained(encoder)
model = model_class.load_from_checkpoint(model_path, encoder=encoder, model_type=model_type)

if torch.cuda.is_available():
    model.cuda()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data.get('sentences', [])

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    inputs = tokenizer(texts, return_tensors='pt', padding=True)
    if torch.cuda.is_available():
        inputs = {key: value.cuda() for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(inputs, return_logits=True).squeeze().tolist()

    if isinstance(outputs, float):
        outputs = [outputs]

    results = [
        {
            'sentence': text,
            'cefr': map_to_cefr(score)
        }
        for text, score in zip(texts, outputs)
    ]

    return jsonify(results)


if __name__ == '__main__':
    # enable cors
    CORS(app)
    app.run(host='0.0.0.0', port=5050)
