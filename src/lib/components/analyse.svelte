<script lang="ts">
      export let memoryTime: [number, string];
      export let computeTime: [number, string];
      export let arithmeticIntensity: number;
      export let hardwareIntensity: number;
      export let totalFLOPs: number;
      export let recommendations: string[];
</script>

<div class="container">
      <div class="card">
            <table class="information">
                  <tbody>
                        <tr>
                              <td>Memory Time</td>
                              <td>{memoryTime[0]} {memoryTime[1]}</td>
                        </tr>
                        <tr>
                              <td>Compute Time</td>
                              <td>{computeTime[0]} {computeTime[1]}</td>
                        </tr>
                        <tr>
                              <td>Arithmetic Intensity</td>
                              <td>~{arithmeticIntensity}</td>
                        </tr>
                        <tr>
                              <td>Hardware Intensity</td>
                              <td>~{hardwareIntensity}</td>
                        </tr>
                        <tr>
                              <td>Total FLOPS</td>
                              <td>{totalFLOPs}</td>
                        </tr>
                  </tbody>
            </table>
            <div class="recommendations">
                  <b>Recommendations</b>
                  <ul>
                        {#each recommendations as point}
                              <li>
                                    {point}
                              </li>
                        {/each}
                  </ul>
            </div>
      </div>
</div>

<style>
      * {
            font-family: "Comic Sans MS", cursive, sans-serif;
      }

      .container {
            padding: 0;
            margin: 0;
            max-width: 100vh;
            max-width: 100vw;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            padding: 2%;
      }

      .card {
            padding: 2%;
            display: flex;
            width: 60%;
            border: 1px solid black;
            flex-direction: column;
            gap: 40px;
            box-shadow: 5px 5px 0px 1px black;
      }

      .information {
            display: flex;
            flex-direction: column;
      }

      td {
            width: 100%;
      }
</style>
