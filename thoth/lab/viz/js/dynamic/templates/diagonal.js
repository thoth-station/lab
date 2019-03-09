/**
 * Dynamic diagonal hierarchical layout
 *
 * @module
 * @author Marek Cermak <macermak@redhat.com>
 */


const margin = { top: 80, right: 20, bottom: 80, left: 20 };

// svg proportions
const width  = $('#notebook-container').width() - 120;  // jupyter notebook margin
const height = 640;

const radius = 11;
const offset = 1.618 * radius;


$(element).empty();  // clear output


const data = d3.csvParse(`$$data`); console.debug("Data: ", data);


let root = d3.stratify()
    .id( d => d.target)
    .parentId( d => d.source)
    (data);

let layout = d3.tree()
    .size([
        width  - margin.right   - margin.left,
        height - margin.top - margin.bottom
    ]);

layout(root);

let svg = d3.select(element.get(0)).append('svg')
    .attr('width', width)
    .attr('height', height)
    .append('g')
    .attr('transform', `translate(0, ${margin.top})`);

let nodes_group = svg.append('g').attr('class', 'nodes'),
    links_group = svg.append('g').attr('class', 'links');


update(root); // initial draw


/**
  * Update diagonal layout
  */
function update(source) {

    console.debug("Root: ", root);
    console.debug("Source: ", source);

    // node circles
    let nodes = nodes_group
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
    let labels = nodes_group
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

    console.debug("Nodes: ", nodes);
    console.debug("Links: ", links);

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

// Collapsing and expanding nodes

/**
  * Handle node click event
  */
function click(d, i) {
    console.debug(`Node ${i} clicked.`);
    console.debug("Node data:", d);

    toggleChildren(this, d, i);  // this will change the root node data too
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

    if (d.children) {
        // collapse
        d.hidden_children = d.children;
        d.children = null;

        d3.select(node).classed('is-collapsed', true);
    } else {
        // expand
        d.children = d.hidden_children;
        d.hidden_children = null;

        d3.select(node).classed('is-collapsed', false);
    }

    // update the layout to compute new node positions
    update(d);
}
