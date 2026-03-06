export async function getGitHubStats() {
  try {
    const response = await fetch(
      "https://api.github.com/repos/anaslimem/cortexadb",
      {
        next: { revalidate: 3600 },
      },
    );

    if (!response.ok) {
      return { stars: 0, forks: 0 };
    }

    const data = await response.json();
    return {
      stars: data.stargazers_count ?? 0,
      forks: data.forks_count ?? 0,
    };
  } catch {
    return { stars: 0, forks: 0 };
  }
}
