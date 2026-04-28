library(circlize)

test <- function(x){
  print(x)
}

generate_ChordDiagram <- function(states, mat, gap, colorlist, transparency, directional, width, height, res, figname){
  mat <- matrix(unlist(mat), ncol = length(states), byrow = TRUE)
  rownames(mat) = states
  colnames(mat) = states
  gap <- as.numeric(unlist(gap))
  
  colorlist <- unlist(colorlist)
  names(colorlist) <- states
  
  

  png(file=paste(figname, '.png', sep=''), units = 'in', width = width, height = height, res = res)
  
  par(cex=0.75) #0.5
  
  circos.par(gap.after = gap, circle.margin = 0.5) #0.75
  chordDiagram(mat, transparency = transparency, grid.col = colorlist,
               directional = directional, annotationTrack = "grid")
  circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index,
                facing = "clockwise", niceFacing = TRUE, adj = c(0., 0.9))
  }, bg.border = NA)
  
  dev.off()
  circos.clear()
}
