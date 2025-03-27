<script lang="ts">
      
</script>

<form>
      <h2>Matrix</h2>
      <p>Mention the shape of two matrices below</p>
      <div style="display: flex; gap: 5px;">
            <input id="A" type="number" placeholder="row">
            <input id="A" type="number" placeholder="col">
      </div>
      <div style="display: flex; gap: 5px;">
            <input id="B" type="number" placeholder="row">
            <input id="B" type="number" placeholder="col">
      </div>
      <select name="format" id="format">
            <option value="bf16">bf16</option>
            <option value="int8">int8</option>
            <option value="fp32">fp32</option>
            <option value="fp16">fp16</option>
      </select>
</form>

<style>
      h2 {
            text-align: center;
      }
      input {
            width: 100%;
            margin: 2% 0 6% 0;
      }
      form {
            background-color: white;
            height: 100%;
            width: 50%;
            padding: 0 2% 2% 2%;
            border: 1px solid grey;
            display: flex;
            flex-direction: column;
            box-shadow: 5px 5px 0px 1px black;
      }
</style>