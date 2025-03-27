<script>
      import Hardware from "$lib/components/hardware.svelte";
      import Matrix from "$lib/components/matrix.svelte";

</script>


<div class="container">
      <div class="main">
            <Matrix/>
            <Hardware/>
      </div>
      <button>
            Analyse
      </button>
</div>

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
            background-color: azure;
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