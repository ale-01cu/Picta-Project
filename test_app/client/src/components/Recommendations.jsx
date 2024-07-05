import { useState } from "react"
import { Link } from "wouter"

export default function Recommendations() {
  const [recs, setRecs] = useState([
    {
      nombre: "aaaa",
      descripcion: "asdjhajsdasd asd ",
      categoria: "a"
    },
    {
      nombre: "aaaa",
      descripcion: "asdjhajsdasd asd ",
      categoria: "a"
    },
  ])

  return (
    <section className="recs-section">
      <div className="recs-container">
        {
          recs.map(e => {
            return (
              <Link to={"detalle/" + e.id} key={e.id} className="rec-card">
                <h2>{e.nombre}</h2>
                <p><span>Descripcion:</span> {e.descripcion}</p>
                <span><span>Categoria:</span> {e.categoria}</span>
              </Link>
            )
          })
        }
      </div>
    </section>
  )
}