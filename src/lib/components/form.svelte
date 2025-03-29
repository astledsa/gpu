<script lang="ts">
      export let data1: number[] = [];
      export let data2: number[] = [];
      export let labels: string[] = [];
      export let titles = ["", "", "", ""];
      export let colors = ["#000000", "#000000", "#000000", "#000000"];
      export let width = 400;
      export let height = 300;
      export let padding = 40;

      $: graphWidth = width - padding * 2;
      $: graphHeight = height - padding * 2;
      function getMaxValue(array: number[]) {
            return Math.max(...array) || 1;
      }

      function getMinValue(array: number[]) {
            return Math.min(...array);
      }

      function scaleY(value: number, max: number, min: number) {
            return graphHeight - ((value - min) / (max - min)) * graphHeight;
      }

      function scaleX(index: number, length: number) {
            return (index / (length - 1)) * graphWidth;
      }

      function createPath(data: number[], color: string) {
            if (data.length === 0) return "";

            const max = getMaxValue(data);
            const min = getMinValue(data);
            const points = data.map((value, i) => {
                  const x = padding + scaleX(i, data.length);
                  const y = padding + scaleY(value, max, min);
                  return `${x},${y}`;
            });

            return `M${points.join(" L")}`;
      }

      function createAreaPath(data: number[], color: string) {
            if (data.length === 0) return "";

            const max = getMaxValue(data);
            const min = getMinValue(data);
            const points = data.map((value, i) => {
                  const x = padding + scaleX(i, data.length);
                  const y = padding + scaleY(value, max, min);
                  return `${x},${y}`;
            });

            return `M${points[0]} L${points.slice(1).join(" L")} L${padding + graphWidth},${padding + graphHeight} L${padding},${padding + graphHeight} Z`;
      }

      function createCircles(data: number[], color: string) {
            if (data.length === 0) return [];

            const max = getMaxValue(data);
            const min = getMinValue(data);
            return data.map((value, i) => {
                  const x = padding + scaleX(i, data.length);
                  const y = padding + scaleY(value, max, min);
                  return { cx: x, cy: y, r: 4, fill: color };
            });
      }
</script>

<div class="graph-container">
      {#each [data1, data2] as data, index}
            <div class="graph-wrapper">
                  <h3>{titles[index]}</h3>
                  <svg class="graph" {width} {height}>
                        <rect {width} {height} fill="#f8f9fa" />

                        <line
                              x1={padding}
                              y1={padding + graphHeight}
                              x2={padding + graphWidth}
                              y2={padding + graphHeight}
                              stroke="#ddd"
                              stroke-width="1"
                        />

                        <line
                              x1={padding}
                              y1={padding}
                              x2={padding}
                              y2={padding + graphHeight}
                              stroke="#ddd"
                              stroke-width="1"
                        />

                        {#each [0.25, 0.5, 0.75, 1] as fraction}
                              <line
                                    x1={padding}
                                    y1={padding + graphHeight * (1 - fraction)}
                                    x2={padding + graphWidth}
                                    y2={padding + graphHeight * (1 - fraction)}
                                    stroke="#eee"
                                    stroke-width="1"
                                    stroke-dasharray="2,2"
                              />
                        {/each}

                        <path
                              d={createAreaPath(data, colors[index])}
                              fill={colors[index] + "33"}
                              stroke="none"
                        />

                        <path
                              d={createPath(data, colors[index])}
                              fill="none"
                              stroke={colors[index]}
                              stroke-width="2"
                        />

                        {#each createCircles(data, colors[index]) as circle}
                              <circle {...circle} />
                        {/each}

                        {#if labels.length === data.length}
                              {#each labels as label, i}
                                    <text
                                          x={padding + scaleX(i, data.length)}
                                          y={height - 10}
                                          text-anchor="middle"
                                          font-size="10"
                                          fill="#666"
                                    >
                                          {label}
                                    </text>
                              {/each}
                        {/if}
                  </svg>
            </div>
      {/each}
</div>

<style>
      .graph-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            border: 1px solid black;
            box-shadow: 5px 5px 0px 1px black;
            background: white;
            padding: 10px;
            width: 60%;
            margin: 2% auto;
      }

      .graph-wrapper {
            padding: 15px;
            background: white;
      }

      .graph-wrapper h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.2em;
            text-align: center;
      }

      .graph {
            display: block;
            margin: 0 auto;
      }
</style>
