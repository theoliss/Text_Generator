// Define the character set used in your model training
const chars = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '+', ',', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '~']; // Add or remove characters as needed
const stoi = {};
chars.forEach((char, index) => stoi[char] = index);
const itos = chars;
const vocabSize = 63;
const sequence_length = 128;

// Encode and decode functions

function encode(str) 
{
    return str.split('').map(char => stoi[char]);
}

function softmax(arr) 
{
    const maxLogit = Math.max(...arr);
    const scaled = arr.map(logit => Math.exp(logit - maxLogit));
    const total = scaled.reduce((acc, val) => acc + val, 0);
    return scaled.map(val => val / total);
}

function sampleIndex(probabilities) {
    const rnd = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probabilities.length; i++) {
        cumSum += probabilities[i];
        if (rnd < cumSum) return i;
    }
    return probabilities.length - 1;
}

function decode(outputIndices) {
    let text = '';

    for (let i = 0; i < outputIndices.length; i += vocabSize) {
        const logits = outputIndices.slice(i, i + vocabSize);
        const probabilities = softmax(logits);
        const randomIndex = weightedRandomSelect(probabilities);
        text += itos[randomIndex];
    }

    return text[text.length-1];
}

function weightedRandomSelect(probabilities) {
    let sum = 0;
    const r = Math.random();
    for (let i = 0; i < probabilities.length; i++) {
        sum += probabilities[i];
        if (r <= sum) return i;
    }
    return probabilities.length - 1; // Return the last index in case random number exceeds sum
}


// Global variable for the ONNX session
let session;

// Load the ONNX model using ONNX Runtime Web
async function loadModel() {
    session = await ort.InferenceSession.create("./model_files/recipes.onnx");
}

async function generateNextChar(context) {
    const encodedInput = encode(context);
    if (encodedInput.length === 0) {console.error('Encoded input is empty');return;}
    const inputTensor = new ort.Tensor("int32", new Int32Array(encodedInput), [1, encodedInput.length]);
    const outputMap = await session.run({'inputs': inputTensor });
    const outputTensor = outputMap['outputs'];
    const output = outputTensor.data;
    const predicted_char = decode(output);

    return predicted_char;
}

// Load the model immediately when the script is loaded
loadModel();

var displayed_texte = document.getElementById('generated_text');
var display_recipe_name = document.getElementById('display_recipe_name');
var generate_button = document.getElementById('generate_btn');
var reset_button = document.getElementById('reset_btn');
var recipe_name = document.getElementById('input_recipe_name_text_box');

generate_button.addEventListener('click', async function() 
{
    let context = recipe_name.value;
    
    if (context == ""){
        context = "fully generated recipe";
    }
    context += " :\n";
    let text_to_show = context;

    document.getElementById("input_paragraph").hidden = true;
    document.getElementById("generate_btn").hidden = true;

    for(let i = 0; i < 1000; i++)
    {
        let memo_char = await generateNextChar(context);
        text_to_show += memo_char;
        if(context.length >= 126)
        {
            context.slice(1);
        }
        context += memo_char;
        document.getElementById('generated_text').innerText = text_to_show
        await new Promise(resolve => setTimeout(resolve,10));
    }
});


reset_button.addEventListener('click', function() 
{
        display_recipe_name.innerHTML = "";
        displayed_texte.innerHTML = "";
        recipe_name.value = "";
        document.getElementById("input_paragraph").hidden = false;
        document.getElementById("generate_btn").hidden = false;
});