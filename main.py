from flask import Flask, request, jsonify
import time

from rlcoder_main import init_rlcoder, load_dataset, run

app = Flask(__name__)

arg = init_rlcoder()
print("RLCoder initialized with arguments:", arg)

bm25, retriver, all_eval_examples = load_dataset(arg)



@app.route('/retrieve', methods=['POST'])
def retrieve_api():
    if request.is_json:
        data = request.get_json()
        left_context = data.get('left_context', '')
    else:
        left_context = request.form.get('left_context', '')
    if not left_context:
        return jsonify({'error': 'left_context is required'}), 400
    
    result = run(arg, left_context, bm25, retriver, all_eval_examples)
    if isinstance(result, list):
        result = [
            {k: v for k, v in item.items() if k != "_type"}
            for item in result
            if isinstance(item, dict)
        ]
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)