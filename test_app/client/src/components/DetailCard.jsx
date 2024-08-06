import { useEffect, useState, useRef } from 'react'
import  getPub from "../apis/getPub"

export const DetailCard = ({id, query}) => {
  const [detail, setDetail] = useState({})
  const prevIdRef = useRef(null)

  useEffect(() => {
    if (prevIdRef.current !== id) {
      prevIdRef.current = id
      getPub(id)
        .then(({data}) => {
          setDetail(data)
        })
    }
  }, [id])

  return (
    <div className='detalle'>
      <h1>{detail.nombre}</h1>
      <h4>{detail.descripcion}</h4>
      <h4>{detail.categoria}</h4>
    </div>
  )
}