import { Route, Switch } from 'wouter'
import { ListCards } from './ListCards'
import { DetailCard } from './DetailCard'
import NotFound from '../pages/NotFound'
import { useState } from 'react'
import Recommendations from './Recommendations'

function Main ({setBuscador, URL_BUSQUEDA, URL_CONTENIDO, setPagination}){
  const [query, setQuery] = useState({})

  return (
    <main>
        <Switch>
          <Route path='/'>
            {params => {
              setQuery('')
              setBuscador("")
              return (
                <div className='home-contain'>
                  <ListCards 
                    url={URL_CONTENIDO} 
                    setPagination={setPagination} 
                    params={params}
                  />
                  <Recommendations/>
                </div>
              )
            }}
          </Route>
          <Route path='/:page'>
            {params => {
              setQuery('')
              setBuscador("")
              return (
                <div className='home-contain'>
                  <ListCards 
                    url={URL_CONTENIDO} 
                    setPagination={setPagination} 
                    params={params}
                  />
                  <Recommendations/>
                </div>
              )
            }}
          </Route>

          <Route path='/detalle/:id'>
            {params => <DetailCard id={params.id} query={query}/>}
          </Route>
          
          <Route path='/result/:busqueda'>
            {params => {
              setQuery(params.busqueda.replaceAll("%20", " "))
              setBuscador(params.busqueda)
              return <ListCards url={URL_BUSQUEDA + params.busqueda} setPagination={setPagination}/>
            }}
          </Route>
          <Route path='' component={NotFound}/>
        </Switch>
    </main>
  )
}

export default Main