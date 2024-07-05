import { LIST_PUBS_URL } from './apis'

const listPubs = async (page=1) => {
    const res = await fetch(LIST_PUBS_URL + "?page=" + page, {
        method: 'GET',
        headers: { 'content-type': 'aplication/json' },
    })
    const data = await res.json()
    return {res, data}
}

export default listPubs