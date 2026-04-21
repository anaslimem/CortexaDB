import { NextResponse } from 'next/server';

export const runtime = 'edge';

export async function GET() {
  try {
    const [githubRes, pypiRes] = await Promise.all([
      fetch('https://api.github.com/repos/anaslimem/CortexaDB', {
        headers: {
          'Accept': 'application/vnd.github.v3+json',
          'User-Agent': 'CortexaDB-Docs',
        },
        next: { revalidate: 3600 }
      }),
      fetch('https://libraries.io/api/pypi/cortexadb', {
        next: { revalidate: 3600 }
      }).catch(() => null)
    ]);

    let stars = 39;
    let downloads = '10K+';
    
    if (githubRes.ok) {
      const githubData = await githubRes.json();
      stars = githubData.stargazers_count || 39;
    }

    if (pypiRes && pypiRes.ok) {
      const pypiData = await pypiRes.json();
      if (pypiData.rank) {
        downloads = pypiData.rank > 100 ? `${Math.round(pypiData.rank / 10)}K+` : `${pypiData.rank}+`;
      }
    }

    return NextResponse.json({
      stars,
      downloads,
    });
  } catch (error) {
    return NextResponse.json({
      stars: 39,
      downloads: '10K+',
      cached: true,
    });
  }
}
