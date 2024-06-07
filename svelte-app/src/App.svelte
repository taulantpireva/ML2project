<script>
  import { onMount } from "svelte";
  let imageUrl = "";
  let detectedFoods = [];
  let nutritionalInfo = {};
  let uploadedImage = null;
  let fileName = "";
  let detectionFailed = false;

  async function handleSubmit() {
    const formData = new FormData();
    formData.append("file", uploadedImage);

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    imageUrl = `data:image/jpeg;base64,${data.image}`;
    detectedFoods = JSON.parse(data.results);
    nutritionalInfo = data.nutritional_info;
    uploadedImage = null; // Clear the uploaded file input
    fileName = ""; // Clear the file name

    // Check if detection failed
    detectionFailed = detectedFoods.length === 0;
  }

  function handleFileChange(event) {
    uploadedImage = event.target.files[0];
    fileName = uploadedImage ? uploadedImage.name : "";
    if (uploadedImage) {
      handleSubmit();
    }
  }
</script>

<div class="container mt-5">
  <h1 class="text-center mb-4">Food Detection</h1>
  <div class="text-center mb-4">
    <label for="file-upload" class="custom-file-upload"> Upload Image </label>
    <input type="file" id="file-upload" on:change={handleFileChange} />
  </div>

  {#if fileName}
    <div class="text-center mb-4">
      <p>Uploaded Image: {fileName}</p>
    </div>
  {/if}

  {#if detectionFailed}
    <div class="text-center text-danger mb-4">
      <p>Model could not detect any food</p>
    </div>
  {/if}

  {#if detectedFoods.length > 0}
    <div class="mb-4">
      <h2>Detected Foods</h2>
      <ul class="list-group">
        {#each detectedFoods as food}
          <li class="list-group-item">
            <strong>{food.name.split("\t")[1]}</strong>
            <div>
              Protein: {nutritionalInfo[food.name.split("\t")[1].trim()]
                ?.protein || "N/A"}g
            </div>
            <div>
              Carbs: {nutritionalInfo[food.name.split("\t")[1].trim()]?.carbs ||
                "N/A"}g
            </div>
            <div>
              Fat: {nutritionalInfo[food.name.split("\t")[1].trim()]?.fat ||
                "N/A"}g
            </div>
          </li>
        {/each}
      </ul>
    </div>
  {/if}

  {#if imageUrl}
    <div class="text-center">
      <img
        src={imageUrl}
        alt="Detected Image"
        class="img-fluid"
        style="max-width: 400px; max-height: 400px;"
      />
    </div>
  {/if}
</div>

<style>
  .form-control-file {
    display: inline-block;
    width: auto;
  }

  input[type="file"] {
    display: none;
  }

  .custom-file-upload {
    border: 1px solid #ccc;
    display: inline-block;
    padding: 6px 12px;
    cursor: pointer;
  }
</style>
