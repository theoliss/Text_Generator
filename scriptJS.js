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
    for(let i = 0; i < 40; i++)
    {
            setTimeout(add_text,200 + i * 200);
    }
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