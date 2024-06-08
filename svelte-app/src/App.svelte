<script>
  import { onMount } from "svelte";
  let imageUrl = "";
  let baseImageUrl = "";
  let detectedFoods = [];
  let nutritionalInfo = {};
  let displayedNutritionalInfo = {};
  let uploadedImage = null;
  let fileName = "";
  let detectionFailed = false;
  let servingSize = "normal";

  async function handleSubmit() {
    const formData = new FormData();
    formData.append("file", uploadedImage);

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    imageUrl = `data:image/jpeg;base64,${data.image}`;
    baseImageUrl = `data:image/jpeg;base64,${data.base_image}`;
    detectedFoods = JSON.parse(data.results);
    nutritionalInfo = data.nutritional_info;
    updateDisplayedNutritionalInfo();
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

  function updateDisplayedNutritionalInfo() {
    displayedNutritionalInfo = {};
    for (let food in nutritionalInfo) {
      let factor = 1;
      if (servingSize === "small") {
        factor = 0.8;
      } else if (servingSize === "large") {
        factor = 1.2;
      }
      displayedNutritionalInfo[food] = {
        protein: (nutritionalInfo[food]?.protein || 0) * factor,
        carbs: (nutritionalInfo[food]?.carbs || 0) * factor,
        fat: (nutritionalInfo[food]?.fat || 0) * factor,
      };
    }
  }

  $: if (servingSize) updateDisplayedNutritionalInfo();
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
      <div class="mb-3">
        <h2>Serving Size</h2>
        <label class="ml-2">
          <input
            type="radio"
            name="serving-size"
            value="small"
            bind:group={servingSize}
          />
          Small
        </label>
        <label class="ml-2">
          <input
            type="radio"
            name="serving-size"
            value="normal"
            bind:group={servingSize}
          />
          Normal
        </label>
        <label class="ml-2">
          <input
            type="radio"
            name="serving-size"
            value="large"
            bind:group={servingSize}
          />
          Large
        </label>
      </div>
      <h2>Detected Food(s)</h2>
      <ul class="list-group">
        {#each detectedFoods as food}
          <li class="list-group-item">
            <strong>{food.name.split("\t")[1]}</strong>
            <div>
              Protein: {displayedNutritionalInfo[
                food.name.split("\t")[1].trim()
              ]?.protein.toFixed(0) || "N/A"}g
            </div>
            <div>
              Carbs: {displayedNutritionalInfo[
                food.name.split("\t")[1].trim()
              ]?.carbs.toFixed(0) || "N/A"}g
            </div>
            <div>
              Fat: {displayedNutritionalInfo[
                food.name.split("\t")[1].trim()
              ]?.fat.toFixed(0) || "N/A"}g
            </div>
            <div>
              Calories:
              {(
                (displayedNutritionalInfo[food.name.split("\t")[1].trim()]
                  ?.protein || 0) *
                  4 +
                (displayedNutritionalInfo[food.name.split("\t")[1].trim()]
                  ?.carbs || 0) *
                  4 +
                (displayedNutritionalInfo[food.name.split("\t")[1].trim()]
                  ?.fat || 0) *
                  9
              ).toFixed(0)}
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

  {#if baseImageUrl}
    <div class="text-center mt-4">
      <h2>Base model yolov5s prediction</h2>
      <img
        src={baseImageUrl}
        alt="Base Detected Image"
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

  .ml-2 {
    margin-left: 0.5rem;
  }
</style>
