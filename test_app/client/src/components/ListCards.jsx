import {useEffect, useState} from 'react'
import { Link } from 'wouter';
import listPubs from '../apis/listPubs'

const URL_DETALLE = '/detalle/'

export const ListCards = ({url, setPagination, params}) => {
  const [cards, setCards] = useState([]);

  useEffect(() => {2
    listPubs(params.page).then(({res, data}) => {

      if (data.pubs) {
        setPagination({'next': data.next, 'previous': data.previous})
        setCards(data.pubs)
      }
      else {
        setPagination({'next': "", 'previous': ""})
        setCards(data.pubs)
      }
    })
  }, [url, setPagination, params.page]);

  return (
    <>
      {/* {
        typeof(cards[cards.length - 1]) === 'number' && <h4 className='tiempo'>Se encontraron {cards.length - 1} resultados en {cards.pop()} segundos.</h4>
      } */}

      <div className="grilla">
        {
          cards.map(e => {
            return (
              <Link to={URL_DETALLE + e.id} key={e.id} className="card">
                <h2>{e.nombre}</h2>
                <p><span>Descripcion:</span> {e.descripcion}</p>
                <span><span>Categoria:</span> {e.categoria}</span>
              </Link>
            )
          })
        }
      </div>
    </>
  )
}

