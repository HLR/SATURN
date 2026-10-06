// Inline Plotly snippets for the per-sample debug report (string.Template placeholders).
// @@block score_bar
(function() {
    var data = [{
        type: "bar", orientation: "h",
        y: $labels_json,
        x: $x_json,
        marker: {color: $colors_json},
        text: $text_json,
        textposition: "outside"
    }];
    var layout = {
        margin: {l: 200, r: 60, t: 30, b: 30},
        xaxis: {title: "Score", range: [0, $xmax]},
        yaxis: {autorange: "reversed"}
    };
    Plotly.newPlot("$div_id", data, layout, {responsive: true});
})();
// @@block heatmap
(function() {
    var data = [{
        type: "heatmap",
        z: $z_json,
        x: $labels_json,
        y: $labels_json,
        colorscale: "Viridis",
        showscale: true
    }];
    var layout = {
        margin: {l: 200, r: 60, t: 30, b: 120},
        xaxis: {tickangle: -45},
        title: "Pairwise scores"
    };
    Plotly.newPlot("$div_id", data, layout, {responsive: true});
})();
// @@block scene3d
(function() {
    var traces = $traces_json;
    var layout = {
        scene: {xaxis: {title: "X (right)"}, yaxis: {title: "-Z (depth)"}, zaxis: {title: "Y (up)"},
                 aspectmode: "data"},
        margin: {l: 10, r: 10, t: 30, b: 10},
        legend: {x: 0.01, y: 0.99}
    };
    Plotly.newPlot("$div_id", traces, layout, {responsive: true});
})();
