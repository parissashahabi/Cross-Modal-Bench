"""
Scene Graph data structures for representing visual scenes.

Following the paper's definition:
- SceneGraph G = {V, E}
- V = {v_1, v_2, ..., v_n} where v_i = {o_i, d_i, attr_i}
- E = {e_1, e_2, ..., e_m} where e_j = {v_j, u_j, r_j}
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class Node:
    """
    Represents an object node in the scene graph.
    
    Attributes:
        id: Unique identifier for this node
        object_class: Object category (e.g., "car", "person")
        description: Optional detailed description
        attributes: Dictionary of attributes (e.g., {"color": "red", "size": "large"})
    """
    id: str
    object_class: str
    description: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary format."""
        return {
            "id": self.id,
            "object": self.object_class,
            "description": self.description,
            "attributes": self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary."""
        return cls(
            id=data.get("id", ""),
            object_class=data.get("object", ""),
            description=data.get("description"),
            attributes=data.get("attributes", {})
        )


@dataclass
class Edge:
    """
    Represents a relationship edge between two nodes.
    
    Attributes:
        subject_id: ID of the subject node
        object_id: ID of the object node
        relation: Type of relationship (e.g., "on", "holding", "next to")
    """
    subject_id: str
    object_id: str
    relation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary format."""
        return {
            "subject": self.subject_id,
            "object": self.object_id,
            "relation": self.relation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        """Create edge from dictionary."""
        return cls(
            subject_id=data.get("subject", ""),
            object_id=data.get("object", ""),
            relation=data.get("relation", "")
        )


@dataclass
class SceneGraph:
    """
    Complete scene graph representation.
    
    Attributes:
        nodes: List of object nodes (V)
        edges: List of relationship edges (E)
        metadata: Optional metadata (image path, timestamp, etc.)
    """
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the scene graph."""
        self.nodes.append(node)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the scene graph."""
        self.edges.append(edge)
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene graph to dictionary format."""
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert scene graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_path: str) -> None:
        """Save scene graph to JSON file."""
        with open(output_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneGraph':
        """Create scene graph from dictionary."""
        nodes = [Node.from_dict(n) for n in data.get("nodes", [])]
        edges = [Edge.from_dict(e) for e in data.get("edges", [])]
        metadata = data.get("metadata", {})
        return cls(nodes=nodes, edges=edges, metadata=metadata)
    
    @classmethod
    def load(cls, json_path: str) -> 'SceneGraph':
        """Load scene graph from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation of the scene graph."""
        return f"SceneGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
