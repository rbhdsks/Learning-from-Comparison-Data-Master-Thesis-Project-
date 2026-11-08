% To do:
% Fix page number so is right aligned. Is a fudge at the moment
% Check all sizes. Are just done by eye at the moment
% Fix \uomtitlespace\ title spacing. Should be automatic
% Check figure spacing. Seems doesn't add any above/below by default
% Update code to listings or similar when possible
% Check more style elements!


%% Class definition
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uom_ltx_talk_casson}[2025/05/23 - Alex Casson - University of Manchester ltx-talk format]
\LoadClass{ltx-talk}



%% Load packages required
\RequirePackage{xcolor}
\RequirePackage{geometry}
\RequirePackage{lua-unicode-math}
\RequirePackage{booktabs}
  \newcommand{\otoprule}{\midrule[\heavyrulewidth]}
\RequirePackage{fancyvrb}



%% Set colours
\definecolor{uom_purple}{RGB}{109, 0, 157}
\colorlet{uom_purple_light}{uom_purple!90}
\definecolor{uom_white}{RGB}{255, 255, 255}
\definecolor{uom_black}{RGB}{0, 0, 0}



%% Process class options
\DeclareOption{purple}{
  \colorlet{uom_background_colour}{uom_purple_light}
  \colorlet{uom_text_colour}{uom_white}
  \newcommand{\uomlogo}{\includegraphics[alt={University of Manchester logo}, width = 0.75\paperwidth, keepaspectratio=true]{images/uom_logo_white.pdf}}
}

\DeclareOption{white}{
  \colorlet{uom_background_colour}{uom_white}
  \colorlet{uom_text_colour}{uom_black}
  \newcommand{\uomlogo}{\includegraphics[alt={University of Manchester logo}, width = 0.15\paperwidth, keepaspectratio=true]{images/uom_logo_colour.pdf}}
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ltx-talk}}
\ProcessOptions\relax



%% Set slide size
\geometry{
  top = 0.085\pageheight,
  headsep = 0.085\pageheight,
}



%% Set main theme
\pagecolor{uom_background_colour}
\color{uom_text_colour}
\renewcommand{\labelitemi}{\textcolor{uom_text_colour}{\textbullet}}
\setmainfont[Ligatures=TeX]{Arial}
\setmathfont{XITS Math}
\AddToHook{begindocument}{%
  \fontsize{14pt}{18pt}\selectfont
}
\SetKeys[talk/frame]{vertical-alignment = top}



%% Format header of each slide. This is empty
\EditInstance{header}{std}{
    background-color = uom_background_colour,
    color = uom_text_colour,
}



%% Format footer. Here just shows slide number
\EditInstance{footer}{std}{
  background-color = uom_background_colour,
  color            = uom_text_colour,
  separator        = \hfill,
  element-order    = {framenumber},
  right-hspace     = 3mm,
  left-hspace      = \paperwidth-7mm, % can't right align at present, so this is somewhat of a hack
}



%% Add logo to every slide
\AddToHook{shipout/foreground}{%
  \put(0.05\paperwidth, -0.2\paperheight){%
    \uomlogo
  }%
}



%% Format title page
\newcommand{\uomdots}{\par\normalfont\makebox[\pagewidth-5em]{\dotfill}} % need to work out actual width

\EditInstance{titlepage-element}{title}{
  color = uom_text_colour,
  font  = \huge\bfseries\raggedright,
  before-skip = 2em,
  after-skip = 0em,
}

\EditInstance{titlepage-element}{subtitle}{
  color = uom_text_colour,
  font  = \uomdots\raggedright\par\Large\raggedright\vspace{0.5em},
  before-skip = 0em,
  after-skip = 1em,
}

\EditInstance{titlepage-element}{author}{
  color = uom_text_colour,
  font  = \Large\raggedright,
  before-skip = 1em,
  after-skip = 1em,
}



%% Format slide title
\EditInstance{frametitle}{header}{
  font = \hspace{0.17\pagewidth}\bfseries\huge
}



%% Tagging options
\tagpdfsetup{math/alt/use} % fix for Canvas equation detection, otherwise Canvas ally doesn't detect equations correctly. Likely not needed in future
