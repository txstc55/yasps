(() => {
  const headings = [...document.querySelectorAll("main h2[id], main h3[id]")];
  const root = document.documentElement;
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const linksById = new Map();

  const addSectionMenu = (pageLink) => {
    if (!pageLink || headings.length === 0) return;

    const list = document.createElement("ol");
    list.className = "section-nav";
    list.setAttribute("aria-label", "On this page");

    for (const heading of headings) {
      const item = document.createElement("li");
      const link = document.createElement("a");

      item.className = `section-nav-level-${heading.tagName.slice(1)}`;
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      item.append(link);
      list.append(item);

      const links = linksById.get(heading.id) || [];
      links.push(link);
      linksById.set(heading.id, links);
    }

    pageLink.insertAdjacentElement("afterend", list);
  };

  addSectionMenu(document.querySelector(".sidebar [aria-current='page']"));

  const mobileMenu = document.querySelector(".mobile-nav > div");
  let mobilePageLink = document.querySelector(
    ".mobile-nav [aria-current='page']"
  );
  if (!mobilePageLink && mobileMenu && headings.length > 0) {
    mobilePageLink = document.createElement("a");
    mobilePageLink.href = location.pathname;
    mobilePageLink.textContent = document.querySelector("h1")?.textContent;
    mobilePageLink.setAttribute("aria-current", "page");
    mobileMenu.prepend(mobilePageLink);
  }
  addSectionMenu(mobilePageLink);

  let currentId = "";
  let scheduled = false;

  const setCurrent = (id) => {
    if (id === currentId) return;

    for (const link of linksById.get(currentId) || []) {
      link.removeAttribute("aria-current");
    }
    for (const link of linksById.get(id) || []) {
      link.setAttribute("aria-current", "location");
    }

    currentId = id;
    const desktopLink = (linksById.get(id) || [])[0];
    if (desktopLink?.offsetParent) {
      desktopLink.scrollIntoView({ block: "nearest" });
    }
  };

  const update = () => {
    scheduled = false;
    const rate = reducedMotion.matches ? 0 : 0.15;
    root.style.setProperty("--dot-offset", `${-scrollY * rate}px`);

    if (headings.length === 0) return;
    const readingLine = Math.min(innerHeight * 0.25, 180);
    let current = headings[0];

    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > readingLine) break;
      current = heading;
    }
    if (scrollY + innerHeight >= document.documentElement.scrollHeight - 2) {
      current = headings[headings.length - 1];
    }
    setCurrent(current.id);
  };

  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(update);
  };

  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  reducedMotion.addEventListener("change", schedule);
  schedule();
})();
