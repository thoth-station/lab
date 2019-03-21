/**
 * Static diagonal hierarchical layout
 *
 * @module
 * @author Marek Cermak <macermak@redhat.com>
 *
 * Require template variables:
 * @param data: hierarchical data in CSV format
 * @param layout: layout to use, one of {'tree', 'cluster'} [default = 'tree']
 */


const margin = { top: 80, right: 20, bottom: 80, left: 20 };

// svg proportions
const width  = $(element).width();
const height = 640;

const radius = 11;
const offset = 1.618 * radius;


$(element).empty();  // clear output


const data = d3.csvParse(`$$data`); console.debug('Data: ', data);

let root = d3.stratify()
    .id( d => d.target)
    .parentId( d => d.source)
    (data);

let layout = d3.$$layout()
    .size([
        width  - margin.right - margin.left,
        height - margin.top   - margin.bottom
    ]);

layout(root); console.debug("Root: ", root);

let svg = d3.select(element.get(0)).append('svg')
    .attr('width', width)
    .attr('height', height);

let g = svg.append('g')
    .attr('transform', `translate(0, ${margin.top})`);

let nodes = g.append('g').attr('class', 'nodes'),
    links = g.append('g').attr('class', 'links');

// node circles
nodes
    .selectAll('circle.node')
    .data(root.descendants())
    .enter()
    .append('circle')
    .attr('class', 'node')
    .attr('cx', d => d.x )
    .attr('cy', d => d.y )
    .attr('r', radius)
    .attr('fill', d => d.data.color );

// node text
nodes
    .selectAll('text.node')
    .data(root.descendants())
    .enter()
    .append('text')
    .attr('class', 'node')
    .attr('x', d => d.x )
    .attr('y', d => d.y )
    .attr('dx', d => d.children ? 1.25 * offset : '' )
    .attr('dy', d => d.children ? '.25em' : radius + 1.25 * offset )
    .attr('text-anchor', d => d.children ? 'right' : 'middle')
    .text( d => d.id /* d.data.name */ );

links
    .selectAll('line.link')
    .data(root.links())
    .enter()
    .append('line')
    .attr('class', 'link')
    .attr('x1', d => d.source.x )
    .attr('y1', d => d.source.y + offset )
    .attr('x2', d => d.target.x )
    .attr('y2', d => d.target.y - offset );

console.debug("Nodes: ", nodes);
console.debug("Links: ", links);
