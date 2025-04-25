#let overlay(img, color) = layout(bounds => {
  let size = measure(img, ..bounds)
  img
  place(top + left, block(..size, fill: color))
})

#let oist_header = image("assets/oist_logo_full_header.svg", fit: "contain", width: 100%)
#set page(
  paper: "a4",
  columns: 1,
  margin: (top: 2.25cm, bottom: 1.5cm, left: 1.5cm, right: 1.5cm),
  header: context {
    if counter(page).get().at(0) == 1 [
      #align(center)[
      #overlay(oist_header, white.transparentize(40%))
      ]
    ]
  },
  footer: context[
    #line(length: 100%)
    #v(-0.5em)
    #h(1fr) #counter(page).display("1") #h(1fr)
  ],
  header-ascent: 0em,
  footer-descent: 0.5em,
)
#set par(justify: true)

#place(
  top + center,
  scope: "parent",
  float: true,
  [
    #text(1.4em, fill: rgb("#8b0000"), weight: "bold")[
      A313 Cognitive Neurorobotics: Project Report
    ]\
    #text(1.2em, weight: "bold")[
      Investigating Adaptive Chunk-based Composition in RNNs
    ]
  ],
)

*Student Name:* Pranshu Malik\
*Date of Submission:* #datetime.today().display("[day] [month repr:long] [year]")\

= Section 1
#lorem(500)

= Section 2
#lorem(500)

#bibliography("refs.bib", title: "References")