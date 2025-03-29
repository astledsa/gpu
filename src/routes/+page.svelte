<script lang="ts">
      import {
            dtype,
            operation,
            type matinfo,
      } from "$lib/types";
      import { H100 } from "$lib/classes";
      import Analyse from "$lib/components/analyse.svelte";
      import MatrixElement from "$lib/components/matrix.svelte";
      import HardwareElement from "$lib/components/hardware.svelte";

      let totalFLOPs: number = 0;
      let numdevices: number = 1;
      let machine: string = "H100";
      let format: dtype = dtype.bf16;
      let hardwareIntensity: number = 0;
      let arithmeticIntensity: number = 0;
      let recommendations: string[] = [""];
      let analyseIspressed: boolean = false;
      let aShape: [number, number] = [0, 0];
      let bShape: [number, number] = [0, 0];
      let sharddim: [number, number] = [0, 0];
      let memoryTime: [number, string] = [0, "s"];
      let computeTime: [number, string] = [0, "s"];

      function convertSecondsToSmallerOrLargerUnits(
            seconds: number,
      ): [number, string] {
            if (Math.abs(seconds) >= 86400) {
                  return [Math.round(seconds / 86400), "days"];
            } else if (Math.abs(seconds) >= 3600) {
                  return [Math.round(seconds / 3600), "hrs"];
            } else if (Math.abs(seconds) >= 60) {
                  return [Math.round(seconds / 60), "min"];
            } else if (Math.abs(seconds) >= 1) {
                  return [Math.round(seconds), "s"];
            } else {
                  let nanoseconds = seconds * 1e9;
                  let microseconds = seconds * 1e6;
                  let milliseconds = seconds * 1e3;

                  if (Math.round(milliseconds) !== 0) {
                        return [Math.round(milliseconds), "ms"];
                  } else if (Math.round(microseconds) !== 0) {
                        return [Math.round(microseconds), "μs"];
                  } else {
                        return [Math.round(nanoseconds), "ns"];
                  }
            }
      }

      function analyse() {
            let lhs: matinfo = { shape: aShape, dtype: format };
            let rhs: matinfo = { shape: bShape, dtype: format };

            if (machine == "H100") {
                  let h100: H100 = new H100();
                  let id: string = h100.addMatrix(operation.matmul, lhs, rhs, {
                        dtype: format,
                  });

                  let { m, r, d } = h100.analyze(id);

                  memoryTime = convertSecondsToSmallerOrLargerUnits(
                        m.memoryTime,
                  );
                  computeTime = convertSecondsToSmallerOrLargerUnits(
                        m.computeTime,
                  );
                  arithmeticIntensity = Math.round(m.arithmeticIntensity);
                  hardwareIntensity = Math.round(m.peakHardwareIntensity);
                  totalFLOPs = m.totalFLOPs;
                  recommendations = r;
                  
                  analyseIspressed = true;
            }
      }

</script>

{#if !analyseIspressed}
      <div class="container">
            <div class="main">
                  <MatrixElement {format} {aShape} {bShape} />
                  <HardwareElement {machine} {numdevices} {sharddim} />
            </div>
            <button on:click={() => analyse()}> Analyse </button>
      </div>
{:else}
      <Analyse
            {memoryTime}
            {computeTime}
            {arithmeticIntensity}
            {hardwareIntensity}
            {totalFLOPs}
            {recommendations}
      />
{/if}

<style>
      :global(body) {
            margin: 0;
            padding: 0;
      }
      .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 5%;
            height: 100vh;
            width: 100vw;
      }
      .main {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5%;
            height: fit-content;
            height: fit-content;
      }
      button {
            border: 1px solid black;
            background-color: black;
            color: white;
            font-weight: bold;
            padding: 10px;
      }
      button:hover {
            border: 1px solid black;
            background-color: white;
            color: black;
            cursor: pointer;
            padding: 10px;
      }

      @media only screen and (max-width: 768px) {
            .main {
                  flex-direction: column;
            }
      }
</style>
