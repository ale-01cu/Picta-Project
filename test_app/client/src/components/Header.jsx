import {Buscador} from './Buscador'
import { Link } from 'wouter'

function Header ({buscador}){

  return (
    <header className="App-header">
      <Link to='/' className='btn-inicio'>Inicio</Link>
      <Buscador valueInput={buscador}/>
    </header>
  )
}

export default Header