/**
 * Dynamic diagonal hierarchical layout
 *
 * @module
 * @author Marek Cermak <macermak@redhat.com>
 */

const margin = { top: 80, right: 20, bottom: 80, left: 20 };

// svg proportions
const width  = $(element).width();  // jupyter notebook margin
const height = 640;

const radius = 11;
const offset = 1.618 * radius;

const transition_duration = 700;

const data = d3.csvParse(`$$data`); console.debug("Data: ", data);

$(element).empty();  // clear output


/* Drawing area setup */

let area = d3.select($(element).get(0));

/* Controls */

let controls = area.append('div')
    .attr('height', margin.top)
    .classed('controls', true);

let button_reset = controls.append('a')
    .attr('class', 'button btn-home')
    .append('span')
    .attr('class', 'icon')
    .append('i')
    .attr('class', 'fas fa-home');

let button_reset_tooltip = button_reset.append('span')
    .attr('class', 'tooltip')
    .text("Default view");

/* SVG Canvas */

let zoom = d3.zoom()
    .extent(() => {
        const rect = $(element).get(0).getBoundingClientRect();

        return [[0, 0], [rect.width, rect.height - margin.top]];
    })
    .scaleExtent([1 / 2, 4])
    .on('zoom', () => {
        let x = d3.event.transform.x,
            y = d3.event.transform.y,
            k = d3.event.transform.k;

        g.attr('transform',
               `translate(${x}, ${y + margin.top / 2}) scale(${k})`);
    });

let svg = area.append('svg')
    .style('width', width)
    .style('height', height)
    .call(zoom)
    .call(zoom, d3.zoomIdentity.translate(0, margin.top));

let g = svg.append('g');


// filter group should be first (due to overlay)
let filters = g.append('g').attr('class', 'filters');

// declare globaly for future reference
let nodes  = null,
    labels = null;

let nodes_group  = g.append('g').attr('class', 'nodes'),
    links_group  = g.append('g').attr('class', 'links');

/* Draw */

let root = d3.stratify()
    .id( d => d.target)
    .parentId( d => d.source)
    (data);

let layout = d3.tree()
    .size([
        width  - margin.right - margin.left,
        height - margin.top   - margin.bottom
    ])(root);

/* Control events */

button_reset
    .on('mouseover', () => {
        button_reset_tooltip
            .transition()
            .delay(transition_duration)
            .style('opacity', 1);
    })
    .on('mouseout', () => {
        button_reset_tooltip
            .transition()
            .delay(transition_duration)
            .style('opacity', 0);
    })
    .on('click', resetView);


update(root);  // initial draw

// set focus on root node
focus(null, root, 0);


/**
 * Reset view to the original scale and position
 *
 * @returns {*}
 */
function resetView(delay) {
    let transform = d3.zoomIdentity
        .scale(1)
        .translate(0, 0);

    svg
        .transition().delay(delay || 200)
        .call(zoom.transform, transform);

    return svg;
}

/**
 * Update diagonal layout
 */
function update(source) {
    console.debug("Layout update triggered", arguments);

    // node circles
    nodes = nodes_group
        .selectAll('circle.node')
        .data(root.descendants(), (d) => d.id || (d.id = ++i));

    nodes
        .enter()
        .append('circle')
        .merge(nodes)
        .attr('cx', d => d.x )
        .attr('cy', d => d.y )
        .attr('r', radius)
        .attr('fill', d => d.data.color)
        .classed('node', true)
        .classed('is-leaf', d => !(d.children || d.hidden_children))
        .on('click', click);  // handle click event

    // node text
    labels = nodes_group
        .selectAll('text.node')
        .data(root.descendants());

    labels
        .enter()
        .append('text')
        .merge(labels)
        .attr('class', 'node')
        .attr('x', d => d.x )
        .attr('y', d => d.y )
        .attr('dx', d => d.children ? 1.25 * offset : "" )
        .attr('dy', d => d.children ? '.25em' : radius + 1.25 * offset )
        .attr('text-anchor', d => d.children ? 'right' : 'middle')
        .text( d => d.id /* d.data.name */ );

    // links
    let links = links_group
        .selectAll('line.link')
        .data(root.links());

    links
        .enter()
        .append('line')
        .merge(links)
        .attr('class', 'link')
        .attr('x1', d => d.source.x )
        .attr('y1', d => d.source.y + offset )
        .attr('x2', d => d.target.x )
        .attr('y2', d => d.target.y - offset );

    /* Collapsible */

    // collapsed nodes and links will be put in `exit` state (see https://bost.ocks.org/mike/join/)
    nodes
        .exit()
        .remove();

    labels
        .exit()
        .remove();

    links
        .exit()
        .remove();
}


/**
 * Handle node click event
 */
function click(d, i) {
    console.debug("Node event triggered: \t", click, arguments);

    let node = d3.select(this);
    if ((d.depth === 0) && (node.classed('is-collapsed'))) {
        // in case only root node is displayed, toggle immediately
        toggleChildren(this, d, i);
    }
    else if (!node.classed('is-focused')) {
        focus(this, d, i);
    } else toggleChildren(this, d, i);  // toggle children (if applicable)
}

/**
 * Focus on node by moving it closer to center in the viewport
 *
 * @param node clicked node element
 * @param d {object} Node data
 * @param idx {object} Node index
 */
function focus(node, d, idx) {
    // remove focus from all previous nodes
    d3.selectAll('.is-focused')
        .classed('is-focused', false);

    // compute center of gravity from focus group
    let n  = 0,
        x0 = 0,
        y0 = 0;

    d.each((d) => {
        n  = n + 1;
        x0 = d.x / n + x0 * (1 - 1/n);
        y0 = d.y / n + y0 * (1 - 1/n);
    });

    svg
        .transition()
        .duration(transition_duration)
        .call(zoom.translateTo, x0, y0);

    // highlight focused node by dropping shadow around it
    node = node ? node : nodes[idx];

    d3.select(node)
        .classed('is-focused', true);

    filters.selectAll('circle').remove();

    // drop focus node
    filters
        .append('circle')
        .attr('cx', d.x )
        .attr('cy', d.y )
        .attr('r', offset)
        .classed('focus-node', true);
}

/**
 * Toggle visibility of children nodes
 *
 * @param node clicked node element
 * @param d {object} Node data
 * @param idx {number} Node index
 */
function toggleChildren(node, d, idx) {
    // Check if the node has children
    console.debug(d);

    // ignore leaf nodes
    if (d3.select(node).classed('is-leaf')) return;

    if (d.children) {
        // collapse
        d.hidden_children = d.children;
        d.children = null;

        d3.select(node).classed('is-collapsed', true);

        focus(nodes[0], root, 0);  // give focus back on root node
    } else {
        // expand
        d.children = d.hidden_children;
        d.hidden_children = null;

        d3.select(node).classed('is-collapsed', false);

        focus(node, d, idx);  // focus still on the expanded node
    }

    // update the layout to compute new node positions
    update(d);
}