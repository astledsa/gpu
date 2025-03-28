<script lang="ts">
      import { dtype } from "$lib/types";

      export let format: dtype;
      export let aShape: [number, number];
      export let bShape: [number, number];
</script>

<form>
      <h2>Matrix</h2>
      <p>Mention the shape of two matrices below</p>
      <div style="display: flex; gap: 5px;">
            <input
                  id="A"
                  type="number"
                  placeholder="row"
                  oninput={(e) => {
                        const target = e.target as HTMLInputElement;
                        aShape[0] = parseInt(target.value);
                  }}
            />
            <input
                  id="A"
                  type="number"
                  placeholder="col"
                  oninput={(e) => {
                        const target = e.target as HTMLInputElement;
                        aShape[1] = parseInt(target.value);
                  }}
            />
      </div>
      <div style="display: flex; gap: 5px;">
            <input
                  id="B"
                  type="number"
                  placeholder="row"
                  oninput={(e) => {
                        const target = e.target as HTMLInputElement;
                        bShape[0] = parseInt(target.value);
                  }}
            />
            <input
                  id="B"
                  type="number"
                  placeholder="col"
                  oninput={(e) => {
                        const target = e.target as HTMLInputElement;
                        bShape[1] = parseInt(target.value);
                  }}
            />
      </div>
      <select
            name="format"
            id="format"
            oninput={(e) => {
                  const target = e.target as HTMLInputElement;
                  switch (target.value) {
                        case "bf16":
                              format = dtype.bf16;
                              break;
                        case "fp8":
                              format = dtype.fp8;
                              break;
                        case "fp32":
                              format = dtype.fp32;
                              break;
                        case "int8":
                              format = dtype.int8;
                              break;
                        case "int4":
                              format = dtype.int4;
                              break;
                        default:
                              format = dtype.fp32; 
                              console.warn(`Unknown target value: ${target.value}`);
                              break;
                  }
            }}
      >
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
