import './index.css'
import Header from './components/Header';
import Main from './components/Main';
import Footer from './components/Footer';
import { useEffect, useState } from 'react';
import { LIST_PUBS_URL } from './apis/apis';
import { useParams, useLocation } from 'wouter';

const URL_BUSQUEDA = `http://localhost:8000/api/search/?busqueda=`


function App() {
  const [buscador, setBuscador] = useState("")
  const [pagination, setPagination] = useState({})
  const [urlContenido, setUrlContenido] = useState(LIST_PUBS_URL)
  const { query } = useLocation();

  useEffect(() => {
    const queryParams = new URLSearchParams(query);
    const page = queryParams.get('page');
    setUrlContenido(LIST_PUBS_URL + "?page=" + page)
  }, [query])

  
  return (
    <div>
      <Header buscador={buscador}/>
      <Footer 
        next={pagination.next || ""} 
        previous={pagination.previous || ""} 
        setUrlContenido={setUrlContenido}
      />
      <Main 
        setBuscador={setBuscador} 
        URL_BUSQUEDA={URL_BUSQUEDA} 
        URL_CONTENIDO={urlContenido} 
        setPagination={setPagination}
      />
    </div>
    );
}

export default App;