# ascii-portrait

把一张插画转成可读的字符画像，并生成慢速动效的网页版本。规则、参数、验收在 [SKILL.md](SKILL.md)。

```bash
python scripts/ascii_portrait.py render portrait.jpg art.txt --cols 300
python scripts/ascii_portrait.py preview art.txt art.png --cell 4
python scripts/ascii_portrait.py page art.txt art.html
```

- `render` 出字符网格（正文）
- `preview` 出等宽单元格 PNG，画布等于画面
- `page` 出单文件动效页，图片与脚本全内联

依赖：`pillow`、`numpy`。
