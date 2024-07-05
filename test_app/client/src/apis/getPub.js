import { LIST_PUBS_URL } from "./apis"

const getPub = async (id) => {
    const res = await fetch(LIST_PUBS_URL + id, {
        method: 'GET',
        headers: { 'content-type': 'aplication/json' },
    })
    const data = await res.json()
    return {res, data} 
}

export default getPub