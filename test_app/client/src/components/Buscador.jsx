import {useEffect, useState} from 'react'
import { useLocation } from 'wouter'

export const Buscador = ({valueInput}) => {
  const [buscador , setBuscador] = useState("")
  const [location , setLocation] = useLocation()
  const [focus, setFocus] = useState(false)

  useEffect(() => {
    const inpputFiltrado = valueInput.replaceAll("%20", " ")
    setBuscador(inpputFiltrado)
  }, [valueInput])

  const handleSubmit = evnt => {
    evnt.preventDefault()
    setLocation("/result/" + buscador)
  }

  const handleChange = (e) => {
    valueInput = e.target.value
    setBuscador(e.target.value)
  }

  const handleFocus = e => {
    setFocus(true)
  }

  const handleBlur = e => {
    setFocus(false)
  }

  return (
    <form action='' onSubmit={handleSubmit}>
      <div className={focus ? 'container-input-search focus' : 'container-input-search'}>
        <button className='btn-buscar'>
          <svg fill='white' xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M19.023 16.977a35.13 35.13 0 0 1-1.367-1.384c-.372-.378-.596-.653-.596-.653l-2.8-1.337A6.962 6.962 0 0 0 16 9c0-3.859-3.14-7-7-7S2 5.141 2 9s3.14 7 7 7c1.763 0 3.37-.66 4.603-1.739l1.337 2.8s.275.224.653.596c.387.363.896.854 1.384 1.367l1.358 1.392.604.646 2.121-2.121-.646-.604c-.379-.372-.885-.866-1.391-1.36zM9 14c-2.757 0-5-2.243-5-5s2.243-5 5-5 5 2.243 5 5-2.243 5-5 5z"></path></svg>
        </button>
        <input 
          type="search" 
          name='busqueda' 
          onChange={handleChange} 
          value={buscador}
          onFocus={handleFocus} 
          onBlur={handleBlur}
          placeholder="Buscar..."
          className='input-search'
        />
      </div>
    </form>
  )
}