python -m venv venv
source bin/activate (for linux)
venv\Scripts\activate (for windows}

pip install -r requirements.txt

uvicorn main:app --reload

Steps:
Add a transcript in db for which meeting you want the summary for

Like in my case it's for meeting_id=20

paste following curl with your bearer token


curl --location 'http://localhost:8081/summary/generate/20' \
--header 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOjEsInN1YiI6InRhaW1vdXJoMDVAZ21haWwuY29tIiwiaWF0IjoxNzc0NTU0MjUwLCJleHAiOjE3NzQ1NTc4NTB9.uo3zkUnVJ1M8eV5XG47oqmJiwZvDEmp04-2HfABgfFk' \
--header 'Content-Type: application/json' \
--data '{}'



This request is towards java backend instead of python-rag
the commuincation between rag and java backend is implmeneted already. No need to go in its detail..
