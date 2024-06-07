<script>
    import { onMount } from 'svelte';
    import axios from 'axios';

    let selectedFile = null;
    let imageUrl = '';
    let results = null;

    const handleFileUpload = async () => {
        if (!selectedFile) {
            alert('Please select a file!');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await axios.post('http://localhost:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            results = response.data.results;
            imageUrl = `data:image/jpeg;base64,${response.data.image}`;
        } catch (error) {
            console.error('Error uploading the file:', error);
        }
    };
</script>

<style>
    img {
        max-width: 100%;
    }
</style>

<h1>Upload an Image for Object Detection</h1>
<input type="file" on:change="{e => selectedFile = e.target.files[0]}" />
<button on:click="{handleFileUpload}">Upload</button>

{#if imageUrl}
    <h2>Detected Objects</h2>
    <img src="{imageUrl}" alt="Detected objects" />
    <pre>{JSON.stringify(results, null, 2)}</pre>
{/if}
