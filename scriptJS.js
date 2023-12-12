// Define the character set used in your model training
const chars = ['%', '$', ' ', 'i', 'f', '&', '#', 'e', '8', 'w', 'g', 'o', "'", '2', '[', 's', 'q', '!', 'k', 'b', 'h', 'z', 'l', '1', '+', '3', '?', '~', ')', 't', '5', 'c', ':', 'n', 'm', 'p', '(', '/', 'j', ']', 'y', 'a', ',', 'u', '"', '0', '6', '7', '_', '@', '{', '`', '4', 'x', '^', '-', 'r', 'd', '9', 'v', '\\', '\n', '}']; // Add or remove characters as needed
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

function decodeLogits(logits) {
    const probabilities = softmax(logits);
    const index = sampleIndex(probabilities);
    return itos[index];
}

function decode(outputIndices) {
    // Assuming each position in outputIndices corresponds to a set of logits for a character
    let text = '';

    for (let i = 0; i < outputIndices.length; i += vocabSize) {
        const logits = outputIndices.slice(i, i + vocabSize);
        const probabilities = softmax(logits);
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        text += itos[maxIndex];
    }

    return text;
}


// Global variable for the ONNX session
let session;

// Load the ONNX model using ONNX Runtime Web
async function loadModel() {
    session = await ort.InferenceSession.create("./model_files/recipes.onnx");
}

// Function to generate text based on the user input
async function generateText() {

    if (!session) {
        console.error('Model not loaded yet');
        return;
    }

    const encodedInput = encode("test");

    if (encodedInput.length === 0) {
        console.error('Encoded input is empty');
        return;
    }

    // Preprocess encodedInput to match the shape [64, 256]
    // This might involve padding or truncating the input
    const paddedInput = new Array(sequence_length).fill(0);  // Filling with zeros (or any appropriate padding value)
    for (let i = 0; i < Math.min(encodedInput.length, sequence_length); i++) {
        paddedInput[i] = encodedInput[i];
    }

    // Create the input tensor with the correct shape
    const inputTensor = new ort.Tensor("int32", new Int32Array(paddedInput), [1, sequence_length]);
    // Run the model
    try {
        const feeds = { 'inputs': inputTensor };
        const outputMap = await session.run(feeds);
        const outputTensor = outputMap['outputs'];
        const outputIndices = Array.from(outputTensor.data);
        const generatedText = decode(outputIndices);
        document.getElementById('generated_text').innerHTML = generatedText;
   
    } 
    catch (error) {
        console.error('Error during model run:', error);
    }
}

// Load the model immediately when the script is loaded
loadModel();
















var word_to_add = " a word";
var generate = true;

var displayed_texte = document.getElementById('generated_text');
var display_recipe_name = document.getElementById('display_recipe_name');
var generate_button = document.getElementById('generate_btn');
var reset_button = document.getElementById('reset_btn');
var recipe_name = document.getElementById('input_recipe_name_text_box');


function add_text()
{
    if(generate)
        {
            displayed_texte.innerHTML = displayed_texte.innerHTML + '\n' + word_to_add ;
        }
}


generate_button.addEventListener('click', function() 
{
    if (recipe_name.value == ""){
        display_recipe_name.innerHTML = "fully generated recipe :";
    }
    else {
        display_recipe_name.innerHTML = recipe_name.value + " :";
    }
    generate = true;
    document.getElementById("input_paragraph").hidden = true;
    document.getElementById("generate_btn").hidden = true;

    generateText();
});


reset_button.addEventListener('click', function() 
{
        display_recipe_name.innerHTML = "";
        displayed_texte.innerHTML = "";
        recipe_name.value = "";
        document.getElementById("input_paragraph").hidden = false;
        document.getElementById("generate_btn").hidden = false;
        generate = false;
});