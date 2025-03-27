<form>
      <h2>Hardware</h2>
      <p>Mention the machine and the number of devices and (if applicable) the sharding dimensions</p>
      <select name="machine" id="machine">
            <option value="h100">H100</option>
            <option value="tpuv5e">TPUv5e</option>
      </select>
      <input type="number" placeholder="devices" defaultValue=1>
      <div style="display: flex; gap: 5px;">
            <input id="A" type="number" placeholder="sharding dimension left" defaultValue=0>
            <input id="A" type="number" placeholder="sharding dimension right" defaultValue=0>
      </div>
</form>

<style>
      h2 {
            text-align: center;
      }
      input {
            width: 98%;
            margin: 2% 0 6% 0;
      }
      form {
            height: 100%;
            width: 50%;
            background-color: white;
            padding: 0 2% 2% 2%;
            border: 1px solid grey;
            display: flex;
            flex-direction: column;
            box-shadow: 5px 5px 0px 1px black;
      }
</style>


