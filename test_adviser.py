from backend.adviser import AgricultureAdviser


adviser = AgricultureAdviser()


question = "What are the major pests of rice?"


result = adviser.answer(
    question
)


print("\n" + "=" * 60)
print("AGRICULTURE ADVISER ANSWER")
print("=" * 60)

print(result["answer"])


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for source in result["sources"]:

    print(
        f"{source['source']} "
        f"- Page {source['page']}"
    )