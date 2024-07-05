import {Link} from 'wouter'
import { LIST_PUBS_URL } from '../apis/apis'

const Footer = ({next = '', previous = '', setUrlContenido}) => {

  const handleClickAnterior = e => {
    setUrlContenido((prev) => prev+ "?page=" + previous)
  }

  const handleClickSiguiente = e => {
    setUrlContenido((prev) => prev+ "?page=" + next)
  }

  return <footer>

    <Link 
      to={"/" + previous} 
      disabled={previous === ''} 
      className='btn-previous'
    >
        Anterior
      </Link>
    
    <Link 
      to={"/" + next} 
      disabled={next === ''} 
      className='btn-next'
    >
      Siguiente
    </Link>
  
  </footer>
  

  
}

export default Footer